from keras.models import Model 
import keras.layers as KL
import keras.backend as K
import keras.engine as KE
import tensorflow as tf 
import numpy as np 
from keras.utils.vis_utils import plot_model
import keras
import utils
import proposal_func
import detection_target_fixed

##########################################################################
#
#                          Resnet
#
##########################################################################

def building_block(filters,block):
    # filters filters的个数    最后输出的channel 是filter的4倍
    # block 如果是0 代表是block1 其他为block2.
    #   block ：  共同特点 都经历fiter 是 1，1，4
    #   block1 ： 特点是 跳层要经历一次卷积+归一化  channel会变成输入filter 的4倍
    #   block2 ： 特点是 跳层直接和输入点点相加
    if block != 0:
        stride = 1
    else:
        stride = 2
    def f(x):
        y = KL.Conv2D(filters,(1,1),strides=stride)(x)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(filters,(3,3),padding='same')(y)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(4 * filters,(1,1))(y)
        y = KL.BatchNormalization(axis=3)(y)
        
        if block == 0:
            shortcut = KL.Conv2D(4 * filters,(1,1),strides=stride)(x)
            shortcut = KL.BatchNormalization(axis=3)(shortcut)
        else:
            shortcut = x
        
        y = KL.Add()([y,shortcut])
        y = KL.Activation('relu')(y)
        return y
    return f

def resnet_feature_extractor(inputs):
    # 构建resnet网络提取特证 网络结构图 看 model_resnet_extractor.jpg
    x = KL.Conv2D(64,(3,3),padding='same')(inputs)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.Activation("relu")(x)

    filter = 64
    blocks = [3,6,4]
    for _ , block in enumerate(blocks):
        for block_id in range(block):
            x = building_block(filter,block_id)(x)
        filter *= 2
    return x

##########################################################################
#
#                          RPN
#
##########################################################################

def rpn_net(inputs,k=9):
    #构建rpn网络

    # inputs : 特证图 shape(batch_size,8,8,1024)
    # k : 特证图上anchor的个数

    # 返回值： 
    #   rpn分类
    #   rpn分类概率
    #   rpn回归
    shared_map = KL.Conv2D(256,(3,3),padding='same')(inputs) #shape(batch_size,8,8,256)
    shared_map = KL.Activation("linear")(shared_map)

    rpn_class = KL.Conv2D(2*k,(1,1))(shared_map) #shape(batch_size,8,8,2*k)
    rpn_class = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,2)))(rpn_class)
    rpn_class = KL.Activation('linear')(rpn_class) #shape(batch_size,8*8*k,2)
    
    rpn_prob = KL.Activation('softmax')(rpn_class)

    rpn_bbox = KL.Conv2D(4*k,(1,1))(shared_map) #shape(batch_size,8,8,4*k)
    rpn_bbox = KL.Activation('linear')(rpn_bbox)
    rpn_bbox = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,4)))(rpn_bbox) #shape(batch_size,8*8*k,4) 8*8*9 = 576

    return rpn_class,rpn_prob,rpn_bbox

##########################################################################
#
#                          RPN_loss
#
##########################################################################

def rpn_class_loss(rpn_matchs,rpn_class_logits):
    # rpn_matchs 分类真值：这个anchor是否有 前景和背景以及中间项  起值对应 1，-1，0 ； shape (?,8*8*9,1)
    # rpn_logist rpn的预测值;  shape (?,8*8*9,2)

    rpn_matchs = tf.squeeze(rpn_matchs,axis=-1) # 压缩tensor翻遍去index
    indices = tf.where(tf.not_equal(rpn_matchs,0)) # 取出 1 和 -1 的框窜参与计算rpn分类
    anchor_class = K.cast(tf.equal(rpn_matchs,1),tf.int32) # 将非1的值转成0，前景为1 ，后景为0
    anchor_class = tf.gather_nd(anchor_class,indices) #target 

    rpn_class_logits = tf.gather_nd(rpn_class_logits,indices) #提取需要的预测值 # prediction

    loss = K.sparse_categorical_crossentropy(anchor_class,rpn_class_logits,from_logits=True)
    loss = K.switch(tf.size(loss) > 0,K.mean(loss),tf.constant(0.0)) #判断loss是否为零

    return loss

def batch_back(x,counts,num_rows):
    out_puts = []
    for i in range(num_rows):
        out_puts.append(x[i,:counts[i]])
    return tf.concat(out_puts,axis=0)
        

def rpn_bbox_loss(target_bbox,rpn_matchs,rpn_bbox):
    # target_bbox 目标框
    # rpn_matchs 真值是否有目标
    # rpn_bbox 预测框
    rpn_matchs = tf.squeeze(rpn_matchs,-1)
    indices = tf.where(K.equal(rpn_matchs,1))

    rpn_bbox = tf.gather_nd(rpn_bbox,indices)# 从预测框中提取对应位置的框

    batch_counts = K.sum(K.cast(K.equal(rpn_matchs,1),'int32'),axis=1) #统计每个图片中有几个bbox
    target_bbox = batch_back(target_bbox,batch_counts,20) #?
    diff  = K.abs(target_bbox - rpn_bbox)
    less_than_one = K.cast(K.less(diff,1),'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one)*(diff - 0.5) #?
    loss = K.switch(tf.size(loss) > 0,K.mean(loss),tf.constant(0.0))
    return loss

##########################################################################
#
#                          proposal
#
##########################################################################
def anchor_refinement(boxes,deltas):
    boxes = tf.cast(boxes,tf.float32)

    #计算被调节框的中心点和高和宽
    h = boxes[:,2] - boxes[:,0]
    w = boxes[:,3] - boxes[:,1]
    center_y = boxes[:,0] + h/2
    center_x = boxes[:,1] + w/2

    #平移和缩放操作
    center_y += deltas[:,0]*h
    center_x += deltas[:,1]*w
    h *= tf.exp(deltas[:,2])
    w *= tf.exp(deltas[:,3])

    #再转化成 （min_y,min_x,max_y,max_x）
    y1 = center_y - h/2
    x1 = center_x - w/2
    y2 = center_y + h/2
    x2 = center_x + w/2

    boxes = tf.stack([y1,x1,y2,x2],axis=1)
    return boxes

#防止边框超出图片范围
def boxes_clip(boxes, window):
    wy1,wx1,wy2,wx2 = tf.split(window,4)
    y1,x1,y2,x2 = tf.split(boxes,4,axis=1)
    y1 = tf.maximum(tf.minimum(y1,wy2),wy1)
    x1 = tf.maximum(tf.minimum(x1,wx2),wx1)
    y2 = tf.maximum(tf.minimum(y2,wy2),wy1)
    x2 = tf.maximum(tf.minimum(x2,wx2),wx1)

    cliped = tf.concat([y1,x1,y2,x2],axis=1)
    cliped.set_shape((cliped.shape[0], 4))#?
    return cliped

# 分片处理
def batch_slice(inputs,graph_fn,batch_size,names=None):
    if not isinstance(inputs,list):
        inputs = [inputs]
    out_puts = []
    for i in range(batch_size):
        input_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*input_slice)
        if not isinstance(output_slice,(list,tuple)):
            output_slice = [output_slice]
        out_puts.append(output_slice)
    out_puts = list(zip(*out_puts))
    if names is not None:
        result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(out_puts, names)]
    else:
        result = [tf.stack(o,axis=0) for o in out_puts]

    if len(result) == 1:
        result = result[0]
    return result

class proposal(KE.Layer):
    def __init__(self,proposal_count,nms_thresh,anchors,batch_size,config,**kwargs):
        super(proposal,self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.nms_thresh = nms_thresh
        self.batch_size = batch_size
        self.config = config
    def call(self,inputs):
        probs = inputs[0][:,:,1] #shape(batch_size,576,1)
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV,(1,1,4)) #denormalization
        prenms = min(100,self.anchors.shape[0]) #最多取出100个
        idxs = tf.nn.top_k(probs,prenms).indices # 钱top

        #提取相关数据
        probs = batch_slice([probs,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        deltas = batch_slice([deltas,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        anchors = batch_slice([idxs],lambda x : tf.gather(self.anchors,x),self.batch_size) #批次内对应的每组anchor

        refined_boxes = batch_slice([anchors,deltas],lambda x,y:anchor_refinement(x,y),self.batch_size) #调整anchor

        #防止 proposal 的框超出图片区域，剪切一下
        H,W = self.config.image_size[:2]
        windows = np.array([0,0,H,W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x: boxes_clip(x,windows),self.batch_size)

        # 对proposal进行归一化  使用的是图片大小进行归一化的
        normalized_boxes = cliped_boxes / tf.constant([H,W,H,W],dtype=tf.float32)

        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes,scores,self.proposal_count,self.nms_thresh)
            box = tf.gather(normalized_boxes,idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(box)[0],0)
            box = tf.pad(box,[(0,pad_num),(0,0)])# 填充0
            return box
        # 对proposal进行nms 最大值抑制
        proposal_ = batch_slice([normalized_boxes,probs],lambda x,y : nms(x,y),self.batch_size)
        return proposal_
  
        
    def compute_output_shape(self,input_shape):
        return (None,self.proposal_count,4)


##########################################################################
#
#                          DetectionTarget
#
##########################################################################
# 去除非0 的部分
def trim_zeros_graph(x,name=None):
    none_zero = tf.cast(tf.reduce_sum(tf.abs(x),axis=1),tf.bool)
    boxes = tf.boolean_mask(x,none_zero,name=name)
    return boxes,none_zero

#todo 需要再看一边
def overlaps_graph(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1,1),[1,1,tf.shape(boxes2)[0]]),[-1,4])
    b2 = tf.tile(boxes2,[tf.shape(boxes1)[0],1])

    b1_y1,b1_x1,b1_y2,b1_x2 = tf.split(b1,4,axis=1)
    b2_y1,b2_x1,b2_y2,b2_x2 = tf.split(b2,4,axis=1)

    y1 = tf.maximum(b1_y1,b2_y1)
    x1 = tf.maximum(b1_x1,b2_x1)
    y2 = tf.minimum(b1_y2,b2_y2)
    x2 = tf.minimum(b1_x1,b2_x1)

    intersection = tf.maximum((y2-y1),0) * tf.maximum((x2-x1),0)
    union = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) + (b2_y2 - b2_y1)*(b2_x2 - b2_x1) - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

def box_refinement_graph(boxes, gt_box):
    
    boxes = tf.cast(boxes,tf.float32)
    gt_box = tf.cast(gt_box,tf.float32)
    

    height = boxes[:,2] - boxes[:,0]
    weight = boxes[:,3] - boxes[:,1]
    center_y = boxes[:,0] + 0.5 * height
    center_x = boxes[:,1] + 0.5 * weight

    gt_height = gt_box[:,2] - gt_box[:,0]
    gt_weight = gt_box[:,3] - gt_box[:,1]
    gt_center_y = gt_box[:,0] + 0.5 * gt_height
    gt_center_x = gt_box[:,1] + 0.5 * gt_weight

    dy = (gt_center_y - center_y)/height # 为什么会除以proposal的高和宽
    dx = (gt_center_x - center_x)/weight
    dh = tf.log(gt_height/height)
    dw = tf.log(gt_weight/weight)

    deltas = tf.stack([dy,dx,dh,dw],axis=1)
    return deltas

def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
    #提取非0 部分：输入，ptoposal 为了固定长度使用 0进行padding
    proposals,_ = trim_zeros_graph(proposals,name='trim_proposals') 
    gt_bboxes,none_zeros = trim_zeros_graph(gt_bboxes,name='trim_bboxes')
    gt_class_ids = tf.boolean_mask(gt_class_ids,none_zeros)

    #计算每个proposal和每个gt_bboxes的iou 
    #加入有N个proposal 和 M个 gt_bboxes
    overlaps = overlaps_graph(proposals, gt_bboxes) #返回的shape：[N,M]
    max_iouArg = tf.reduce_max(overlaps,axis=1) # 沿着M压缩 取出N个最大值  用来判断哪个proposal 是前景
    max_iouGT = tf.argmax(overlaps,axis=0)# 沿着N压缩  计算出proposal 对应最适应的gt_bboxes

    positive_mask = max_iouArg > 0.5 #大于0.5的为前景
    positive_idxs = tf.where(positive_mask)[:,0] # 前景索引
    negative_idxs = tf.where(max_iouArg < 0.5)[:,0] # 背景索引


    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio) #前景的数量
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    positive_idxs = tf.concat([positive_idxs, max_iouGT], axis=0)
    positive_idxs = tf.unique(positive_idxs)[0] # 前景索引
    
    num_positive = tf.shape(positive_idxs)[0] #前景的数量

    r = 1 / config.num_proposals_ratio
    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive #背景的数量
    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]#背景索引

    positive_rois = tf.gather(proposals,positive_idxs)
    negative_rois = tf.gather(proposals,negative_idxs)

    # 取出前景对应的gt_bbox
    positive_overlap = tf.gather(overlaps,positive_idxs)
    gt_assignment = tf.argmax(positive_overlap,axis=1)
    gt_bboxes = tf.gather(gt_bboxes,gt_assignment)
    gt_class_ids = tf.gather(gt_class_ids,gt_assignment)


    # 计算偏移量
    deltas = box_refinement_graph(positive_rois, gt_bboxes)
    deltas /= config.RPN_BBOX_STD_DEV # 算出来的太小，需要统一增大

    rois = tf.concat([positive_rois, negative_rois], axis=0)

    N = tf.shape(negative_rois)[0]
    P = config.num_proposals_train - tf.shape(rois)[0]
    
    rois = tf.pad(rois,[(0,P),(0,0)])
    gt_class_ids = tf.pad(gt_class_ids, [(0, N+P)])
    deltas = tf.pad(deltas,[(0,N+P),(0,0)])
    gt_bboxes = tf.pad(gt_bboxes,[(0,N+P),(0,0)])
    
    return rois, gt_class_ids, deltas, gt_bboxes

class DetectionTarget(KE.Layer):
    def __init__(self,config,**kwargs):
        super(DetectionTarget,self).__init__()
        self.config = config
    def call(self,inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_bboxes = inputs[2]

        names = ["rois", "target_class_ids", "target_deltas","target_bbox"]

        outputs = batch_slice([proposals,gt_class_ids,gt_bboxes],
                    lambda x,y,z:detection_target_graph(x,y,z,self.config),self.config.batch_size,names)

        return outputs


    def compute_output_shape(self,input_shape):
        return [None, None, None, None]


##########################################################################
#
#                          utils
#
##########################################################################
class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)



##########################################################################
#
#                          fpn_classifiler
#
##########################################################################

def roi_pooling_onebacth(img, rois, num_rois, pool_size, nb_channels):
    img = K.expand_dims(img, 0)
    outputs = []
    for roi_idx in range(num_rois):

        y1 = rois[roi_idx, 0]
        x1 = rois[roi_idx, 1]
        y2 = rois[roi_idx, 2]
        x2 = rois[roi_idx, 3]

        x2 = x1 + K.maximum(1.0,x2-x1)
        y2 = y1 + K.maximum(1.0,y2-y1)

        y1 = K.cast(y1, 'int32')
        x1 = K.cast(x1, 'int32')
        y2 = K.cast(y2, 'int32')
        x2 = K.cast(x2, 'int32')

        rs = tf.image.resize_images(img[:, y1:y2, x1:x2, :], (pool_size, pool_size))
        outputs.append(rs)

    final_output = K.concatenate(outputs, axis=0)
    final_output = K.reshape(final_output, (-1, num_rois, pool_size, pool_size, nb_channels))
    return final_output

class RoiPoolingConvV2(KE.Layer):
    def __init__(self,pool_size,num_rois,batch_size,**kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.batch_size = batch_size

        super(RoiPoolingConvV2,self).__init__(**kwargs)

    def build(self,input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        if self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]
    
    def compute_output_shape(self,input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self,x,mask=None):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        out = batch_slice([img,rois],lambda x,y: roi_pooling_onebacth(x,y,self.num_rois, self.pool_size, self.nb_channels), \
                                batch_size=self.batch_size)
        out = K.reshape(out, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return out

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConvV2, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
  

def fpn_classifiler(feature_map, rois, batch_size, num_rois, pool_size, num_classes):
    x = RoiPoolingConvV2(7,num_rois,batch_size)([feature_map,rois])
    print(feature_map)
    print(x)
    x = KL.TimeDistributed(KL.Conv2D(512, pool_size, padding="valid"),name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),name="fpn_classifier_bn0")(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(512, (1, 1), padding="valid"), name="fpn_classifier_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3), name="fpn_classifier_bn1")(x)
    x = KL.Activation("relu")(x)

    base = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),name="fpn_classifier_squeeze")(x)

    #分类
    class_logits = KL.TimeDistributed(KL.Dense(num_classes),name='fpn_classifier_logits')(base)
    class_prob = KL.TimeDistributed(KL.Activation('softmax'),name="fpn_classifier_prob")(class_logits)

    #回归
    class_fc = KL.TimeDistributed(KL.Dense(4*num_classes,activation='linear'),name="fpn_classifier_fc")(base)
    s = K.int_shape(class_fc)
    class_bbox = KL.Reshape((s[1],num_classes,4), name="fpn_class_deltas")(class_fc)

    return class_logits,class_prob,class_bbox

##########################################################################
#
#                          fpn_loss
#
##########################################################################
def smooth_l1_loss(y_true,y_pred):
    diff = tf.abs(y_true-y_pred)
    less_than_one = K.cast(tf.less(diff,1.0),'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one)(diff - 0.5)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    target_class_ids =K.reshape(target_class_ids,(-1,))
    target_bbox = K.reshape(target_bbox,(-1,4))
    pred_bbox = K.reshape(pred_bbox,(-1,K.int_shape(pred_bbox)[2],4)) #?

    positive_roi_ix = tf.where(target_class_ids > 0)[:,0]
    positive_roi_class_ids = K.cast(tf.gather(target_class_ids,positive_roi_ix),tf.int32)

    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)#?

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices) #?

    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def mrcnn_class_loss_graphV2(target_class_ids, pred_class_logits,active_class_ids, batch_size=20):
    target_class_ids = tf.cast(target_class_ids, 'int64')
    pred_class_ids = tf.argmax(pred_class_logits,axis=2)

    pred_active = batch_slice([active_class_ids, pred_class_ids],lambda x,y:tf.gather(x,y),batch_size=batch_size)#?

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    pred_active = tf.cast(pred_active, tf.float32)

    loss = loss * pred_active


    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss
                
##########################################################################
#
#                          DetectionLayer
#
##########################################################################

def refine_detections(rois, probs, deltas):
    argMax_probs = tf.argmax(probs, axis=1)
    max_probs = tf.reduce_max(probs, axis=1)
    keep_idxs = tf.where(max_probs > 0.5)[:,0]
    idx_y = tf.cast(np.arange(16), tf.int32)
    idx_x = tf.cast(argMax_probs, tf.int32)
    idxs = tf.stack([idx_y, idx_x],axis=1)
    deltas_keep = tf.gather_nd(deltas, idxs)
    refined_rois = anchor_refinement(tf.cast(rois, tf.float32),
                                 tf.cast(deltas_keep * config.RPN_BBOX_STD_DEV, tf.float32))
    rois_ready = tf.gather(refined_rois, keep_idxs)
    class_ids = tf.gather(argMax_probs, keep_idxs)
    class_ids = tf.to_float(class_ids)[..., tf.newaxis]
    detections = tf.concat([rois_ready, class_ids], axis=1)
    gap = tf.maximum(16 - tf.shape(detections)[0],0)
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

### NMS

class DetectionLayer(KE.Layer):

    def __init__(self, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        rois = inputs[0]
        probs = inputs[1]
        deltas = inputs[2]
        
        detections_batch = utils.batch_slice(
            [rois, probs, deltas],
            lambda x, y, z: refine_detections(x, y, z),
            20)
        
        #return tf.reshape(
        #    detections_batch,
        #    [16, 8, -1])
        return detections_batch

    def compute_output_shape(self, input_shape):
        return (None, 8, -1)

##########################################################################
#
#                          faster_rcnn
#
##########################################################################

class FasterRcnn():
    def __init__(self,mode,subnet,config):
        assert mode in ["training","inference"]

        self.mode = mode
        self.subnet = subnet
        self.config = config
        self.keras_model = self.build(self.mode,self.subnet,self.config)
    
    def build(self,mode,sub_net,config):
        assert mode in ["training","inference"]

        h, w = config.image_size[: 2]
        #模型输入
        input_image = KL.Input(shape=[h,w,3],dtype=tf.float32)
        input_bboxes = KL.Input(shape=[None,4],dtype=tf.float32)
        input_class_ids = KL.Input(shape=[None],dtype=tf.int32)
        input_active_ids = KL.Input(shape=[4,], dtype=tf.int32) #?
        input_rpn_match = KL.Input(shape=[None,1],dtype=tf.int32)
        input_rpn_box = KL.Input(shape=[None,4],dtype=tf.float32)

        #归一化gt_bboxes
        imgae_scale = K.cast(tf.stack([h,w,h,w]),tf.float32)
        gt_bboxes = KL.Lambda(lambda x : x/imgae_scale)(input_bboxes)


        feature_map = resnet_feature_extractor(input_image) # 特征提取  输出的shape(batch_size,8,8,1024)
        rpn_class,rpn_prob,rpn_bbox = rpn_net(feature_map,9) # rpn 推荐  shape(batch_size,576,2),shape(batch_size,576,1),shape(batch_size,576,4)                    # todo  

        #获取anchors 
        anchors = utils.anchor_gen(featureMap_size=[8,8],ratios=config.ratios, #todo
                                scales=config.scales,rpn_stride=config.rpn_stride,anchor_stride=config.anchor_stride) # shape(576,4)

        #proposal 提取边框
        proposals = proposal_func.proposal(proposal_count=16,nms_thresh=0.7,anchors=anchors,batch_size=config.batch_size,config=config)([rpn_prob,rpn_bbox])

        if mode == 'training':
            # 将proposal 和 真值 转化成  fpn的训练数据 delta
            target_rois, target_class_ids, target_delta, target_bboxes = detection_target_fixed.DetectionTarget(config,name="proposal_target")([proposals,input_class_ids,gt_bboxes])

            denomrlaize_rois = KL.Lambda(lambda x: 8.0 * x,name="denormalized_rois")(target_rois) #把roi放在特征图的维度上 proposals 是归一化后的数据

            #分类和回归 
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifiler(feature_map, denomrlaize_rois, 
                                                batch_size=config.batch_size, num_rois=21, pool_size=7, num_classes=4)
            
            #rpn_loss
            loss_rpn_match = KL.Lambda(lambda x:rpn_class_loss(*x),name='loss_rpn_match')([input_rpn_match,rpn_prob])
            loss_rpn_bbox = KL.Lambda(lambda x:rpn_bbox_loss(*x),name='loss_rpn_bbox')([input_rpn_box,rpn_bbox])
            #floss 
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="bbox_loss")(
                                                [target_delta, target_class_ids, mrcnn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graphV2(*x), name="mrcnn_class_loss")(
                                        [target_class_ids, mrcnn_class_logits, input_active_ids])

            if sub_net == 'rpn':
                model = Model([input_image,input_bboxes,input_class_ids,input_active_ids,input_rpn_match,input_rpn_box],[feature_map, rpn_class, rpn_prob, rpn_bbox, proposals, target_rois, denomrlaize_rois,target_class_ids,target_delta, target_bboxes, loss_rpn_match, loss_rpn_bbox])
            elif sub_net == 'all':
                model = Model([input_image,input_bboxes,input_class_ids,input_active_ids,input_rpn_match,input_rpn_box],[feature_map, rpn_class, rpn_prob, rpn_bbox, proposals, target_rois, denomrlaize_rois,target_class_ids, target_delta, target_bboxes, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, loss_rpn_match, loss_rpn_bbox, bbox_loss, class_loss])
            
        if mode == "inference":
            denomrlaize_proposals = KL.Lambda(lambda x:8.0*x, name="denormalized_proposals")(proposals)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifiler(feature_map, denomrlaize_proposals, 20, 16, 7, 4)
            detections = DetectionLayer()([proposals, mrcnn_class, mrcnn_bbox])
            
            model = Model([input_image],[detections])

        return model
    
    def compile_(self):
        loss_lay1 = self.keras_model.get_layer("loss_rpn_match").output
        loss_lay2 = self.keras_model.get_layer("loss_rpn_bbox").output
        if self.subnet == "all":
            loss_lay3 = self.keras_model.get_layer("bbox_loss").output
            loss_lay4 = self.keras_model.get_layer("mrcnn_class_loss").output

        self.keras_model.add_loss(tf.reduce_mean(loss_lay1))
        self.keras_model.add_loss(tf.reduce_mean(loss_lay2))
        if self.subnet == "all":
            self.keras_model.add_loss(tf.reduce_mean(loss_lay3))
            self.keras_model.add_loss(tf.reduce_mean(loss_lay4))

        self.keras_model.compile(loss=[None]*len(self.keras_model.output), optimizer=keras.optimizers.SGD(lr=0.00005, momentum=0.9))

        self.keras_model.metrics_names.append("loss_rpn_match")
        self.keras_model.metrics_tensors.append(tf.reduce_mean(loss_lay1, keep_dims=True))

        self.keras_model.metrics_names.append("loss_rpn_bbox")
        self.keras_model.metrics_tensors.append(tf.reduce_mean(loss_lay2, keep_dims=True))

        if self.subnet == "all":
            self.keras_model.metrics_names.append("bbox_loss")
            self.keras_model.metrics_tensors.append(tf.reduce_mean(loss_lay3, keep_dims=True))

            self.keras_model.metrics_names.append("mrcnn_class_loss")
            self.keras_model.metrics_tensors.append(tf.reduce_mean(loss_lay4, keep_dims=True))

    def training(self,trainGen):
        self.compile_()
        self.keras_model.fit_generator(trainGen,steps_per_epoch=20, epochs=100) #todo

    def inference(self,testData):
        assert self.mode == "inference"
        out = self.keras_model.predict(testData)
        return out

    def saveWeights(self,weights_path):
        self.keras_model.save_weights(weights_path)

    def loadWeights(self,weights_path):
        self.keras_model.load_weights(weights_path)

    


if __name__ == "__main__":
    ''' rest_net test
    input_tensor = KL.Input((32,32,256))
    y = building_block(64,3)(input_tensor)
    model = Model([input_tensor],[y])
    model.summary()
    plot_model(model,to_file="model.jpg")
    '''
    '''  resnet_feature_extractor test
    input_tensor = KL.Input((64,64,3))
    y = resnet_feature_extractor(input_tensor)
    model = Model([input_tensor],[y])
    model.summary()
    plot_model(model,to_file="model_resnet_extractor.jpg")
    ''' 

    '''  rpn_net test 
    input_tensor = KL.Input((64,64,3))
    y = resnet_feature_extractor(input_tensor)
    rpn_class,rpn_prob,rpn_bbox = rpn_net(y,9)
    model = Model([input_tensor],[rpn_class,rpn_prob,rpn_bbox])
    model.summary()
    plot_model(model,to_file="model_rpn_net.jpg")
    '''
    '''  loss test
    input_images = KL.Input((64,64,3))
    input_rpn_matchs = KL.Input((None,1))
    input_rpn_bbox = KL.Input((None,4))

    y = resnet_feature_extractor(input_images)
    rpn_class,rpn_prob,rpn_bbox = rpn_net(y,9)

    loss_rpn_matchs = KL.Lambda(lambda x:rpn_class_loss(*x),name="loss_rpn_match")([input_rpn_matchs,rpn_prob])
    loss_rpn_bbox = KL.Lambda(lambda x : rpn_bbox_loss(*x),name="loss_rpn_bbox")([input_rpn_bbox,input_rpn_matchs,rpn_bbox])

    model = Model([input_images,input_rpn_matchs,input_rpn_bbox],[rpn_class,rpn_prob,rpn_bbox,input_rpn_bbox,loss_rpn_bbox,loss_rpn_matchs])

    model.summary()
    plot_model(model,to_file="model_rpn_loss.jpg")
    '''
    ''' train
    input_image = KL.Input(shape=[64,64,3], dtype=tf.float32)
    input_bboxes = KL.Input(shape=[None,4], dtype=tf.float32)
    input_class_ids = KL.Input(shape=[None],dtype=tf.int32)
    input_rpn_match = KL.Input(shape=[None, 1], dtype=tf.int32)
    input_rpn_bbox = KL.Input(shape=[None, 4], dtype=tf.float32)

    y = resnet_feature_extractor(input_image)
    rpn_class,rpn_prob,rpn_bbox = rpn_net(y,9)

    loss_rpn_matchs = KL.Lambda(lambda x:rpn_class_loss(*x),name="loss_rpn_match")([input_rpn_match,rpn_prob])
    loss_rpn_bbox = KL.Lambda(lambda x : rpn_bbox_loss(*x),name="loss_rpn_bbox")([input_rpn_bbox,input_rpn_match,rpn_bbox])

    model = Model([input_image,input_bboxes,input_class_ids,input_rpn_match,input_rpn_bbox],
    [rpn_class,rpn_prob,rpn_bbox,input_rpn_bbox,loss_rpn_bbox,loss_rpn_matchs])

    loss_lay1 = model.get_layer('loss_rpn_match').output
    loss_lay2 = model.get_layer('loss_rpn_bbox').output
    model.add_loss(loss_lay1)
    model.add_loss(loss_lay2)

    model.compile(loss=[None]*len(model.output),optimizer=keras.optimizers.SGD(lr=0.005,momentum=0.9))

    model.metrics_names.append('loss_rpn_match')
    model.metrics_tensors.append(tf.reduce_mean(loss_lay1,keep_dims=True))
    model.metrics_names.append('loss_rpn_bbox')
    model.metrics_tensors.append(tf.reduce_mean(loss_lay2,keep_dims=True))

    #数据gen
    from utils import shapeData as dataSet
    from config import Config

    config = Config()
    dataset = dataSet([64,64], config=config)

    def data_Gen(dataset, num_batch, batch_size, config):
        for _ in range(num_batch):
            images = []
            bboxes = []
            class_ids = []
            rpn_matchs = []
            rpn_bboxes = []
            for i in range(batch_size):
                image, bbox, class_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()
                pad_num = config.max_gt_obj - bbox.shape[0]
                pad_box = np.zeros((pad_num, 4))
                pad_ids = np.zeros((pad_num, 1))
                bbox = np.concatenate([bbox, pad_box], axis=0)
                class_id = np.concatenate([class_id, pad_ids], axis=0)

                images.append(image)
                bboxes.append(bbox)
                class_ids.append(class_id)
                rpn_matchs.append(rpn_match)
                rpn_bboxes.append(rpn_bbox)
            images = np.concatenate(images, 0).reshape(batch_size, config.image_size[0],config.image_size[1] , 3)
            bboxes = np.concatenate(bboxes, 0).reshape(batch_size, -1 , 4)
            class_ids = np.concatenate(class_ids, 0).reshape(batch_size, -1 )
            rpn_matchs = np.concatenate(rpn_matchs, 0).reshape(batch_size, -1 , 1)
            rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1 , 4)
            yield [images, bboxes, class_ids, rpn_matchs, rpn_bboxes],[]

    dataGen = data_Gen(dataset, 35000, 20, config)

    #训练
    his = model.fit_generator(dataGen,steps_per_epoch=20, epochs=1200)
    model.save_weights("model_material.h5")
    '''

    ''' faster-rcnn 测试'''
    from utils import shapeData as dataSet
    from config import Config
    config = Config()
    dataset = dataSet([64,64], config=config)

    model = FasterRcnn(mode="training", subnet="all", config=config)
    '''
    def data_Gen(dataset, num_batch, batch_size, config):
        for _ in range(num_batch):
            images = []
            bboxes = []
            class_ids = []
            rpn_matchs = []
            rpn_bboxes = []
            active_ids = []
            for i in range(batch_size):
                image, bbox, class_id, active_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()
                pad_num = config.max_gt_obj - bbox.shape[0]
                pad_box = np.zeros((pad_num, 4))
                pad_ids = np.zeros((pad_num, 1))
                bbox = np.concatenate([bbox, pad_box], axis=0)
                class_id = np.concatenate([class_id, pad_ids], axis=0)

                images.append(image)
                bboxes.append(bbox)
                class_ids.append(class_id)
                rpn_matchs.append(rpn_match)
                rpn_bboxes.append(rpn_bbox)
                active_ids.append(active_id)
            images = np.concatenate(images, 0).reshape(batch_size, config.image_size[0],config.image_size[1] , 3)
            bboxes = np.concatenate(bboxes, 0).reshape(batch_size, -1 , 4)
            class_ids = np.concatenate(class_ids, 0).reshape(batch_size, -1 )
            active_ids = np.concatenate(active_ids, 0).reshape(batch_size, -1 )
            rpn_matchs = np.concatenate(rpn_matchs, 0).reshape(batch_size, -1 , 1)
            rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1 , 4)
            rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1 , 4)
            yield [images, bboxes, class_ids, active_ids, rpn_matchs, rpn_bboxes],[]

    dataGen = data_Gen(dataset, 35000, 20, config)
    model.training(dataGen)
    '''
    






    
    




    
































