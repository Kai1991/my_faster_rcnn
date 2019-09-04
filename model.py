from keras.models import Model 
import keras.layers as KL
import keras.backend as K
import keras.engine as KE
import tensorflow as tf 
import numpy as np 
from keras.utils.vis_utils import plot_model
import keras


'''
#######################################################################################
def custom_loss1(y_true,y_pred):
    return K.mean(K.abs(y_true - y_pred))

def custom_loss2(y_true,y_pred):
    return K.mean(K.square(y_true - y_pred))



input_tensor1 = KL.Input((32,32,3))
input_tensor2 = KL.Input((4,))
target = KL.Input((2,))

x = KL.BatchNormalization(axis=-1)(input_tensor1)
x = KL.Conv2d(16,(3,3),padding='same') (x)
x= KL.Activation("relu")(x)
x = KL.MaxPool2D(2)(x)
x = KL.Conv2D(32,(3,3),padding='same')(x)
x = KL.Activation('relu')(x)
x = KL.MaxPool2D(2)(x)
x = KL.Flatten()(x)
x = KL.Dense(32)(x)
out2  = KL.Dense(2)(x)


y = KL.Dense(32)(input_tensor2)
out1 = KL.Dense(2)(y)

loss1 = KL.Lambda(lambda x:custom_loss1(*x),name='loss1')([out2,out1])

loss2 = KL.Lambda(lambda x:custom_loss1(*x),name='loss2')([target,out2])

model = Model([input_tensor1,input_tensor2,target],[out1,out2,loss1])

model.summary()

#training

loss_lay1 = model.get_layer("loss1").output
loss_lay2 = model.get_layer("loss2").output

model.add_loss(loss_lay1)
model.add_loss(loss_lay2)

model.compile(optimizer='sgd',loss=[None,None,None,None])


def data_gen(num):
    for i in range(num):
        yield [np.random.normal(1,1,(1,32,32,3)),np.random.normal(1,1,(1,4)),np.random.normal(1,1,(1,2))],[]

data_set = data_gen(100000)
model.fit_generate(data_set,step_per_epoch=100,epochs=20)
############################################################################################
'''

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

def rpn_net(inputs,k):
    #构建rpn网络

    # inputs : 特证图
    # k : 特证图上anchor的个数

    # 返回值： 
    #   rpn分类
    #   rpn分类概率
    #   rpn回归
    shared_map = KL.Conv2D(256,(3,3),padding='same')(inputs)
    shared_map = KL.Activation("linear")(shared_map)

    rpn_class = KL.Conv2D(2*k,(1,1))(shared_map)
    rpn_class = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,2)))(rpn_class)
    rpn_class = KL.Activation('linear')(rpn_class)
    
    rpn_prob = KL.Activation('softmax')(rpn_class)

    rpn_bbox = KL.Conv2D(4*k,(1,1))(shared_map)
    rpn_bbox = KL.Activation('linear')(rpn_bbox)
    rpn_bbox = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,4)))(rpn_bbox)

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

    h = boxes[:,2] - boxes[:,0]
    w = boxes[:,3] - boxes[:,1]
    center_y = boxes[:,0] + h/2
    center_x = boxes[:,1] + w/2

    #平移和缩放操作
    center_y += deltas[:,0]*h
    center_x += deltas[:,1]*w
    h *= tf.exp(deltas[:,2])
    w *= tf.exp(deltas[:,3])

    y1 = center_y - h/2
    x1 = center_x - w/2
    y2 = center_y + h/2
    x2 = center_x + w/2

    boxes = tf.stack([y1,x1,y2,x2],axis=1)
    return boxes

#防止边框超出图片范围
def boxes_clip(boxes, window):
    wy1,wx1,wy2,wx2 = tf.split(window,4)
    y1,x1,y2,x2 = tf.split(boxes,4)
    y1 = tf.maximum(y1,wy1)
    x1 = tf.maximum(x1,wx1)
    y2 = tf.minimum(y2,wy2)
    x2 = tf.minimum(x2,wx2)

    cliped = tf.concat([y1,x1,y2,x2],axis=1)
    cliped.set_shape((cliped.shape[0], 4))#?
    return cliped

# 分片处理
def batch_slice(inputs,graph_fn,batch_size,names=None):
    if not isinstance(inputs):
        inputs = [inputs]
    out_puts = []
    for i in range(batch_size):
        input_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*input_slice)
        if not isinstance(output_slice,(list,tuple)):
            output_slice = [output_slice]
        out_puts.append(output_slice)
    out_puts = list(zip(*out_puts))
    if names is None:
        result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(out_puts, names)]
    else:
        result = [tf.stack(o,axis=0) for o in out_puts]

    if len(result) == 1:
        result = result[0]
    return result

class proposal(KE.Layer):
    def __init__(self,proposal_count,nms_thresh,anchors,batch_size,config,**kwargs):
        super(proposal,self).__init__(kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.nms_thresh = nms_thresh
        self.batch_size = batch_size
        self.config = config
    def call(self,inputs):
        probs = inputs[0][:,:,1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV,(1,1,4))
        prenms = min(100,self.anchors.shape[0])
        idxs = tf.nn.top_k(probs,prenms).indices

        probs = batch_slice([probs,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        deltas = batch_slice([deltas,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        anchors = batch_slice([idxs],lambda x : tf.gather(self.anchors,idxs),self.batch_size)

        refined_boxes = batch_slice([anchors,deltas],lambda x,y:anchor_refinement(x,y),self.batch_size)

        H,W = self.config.image_size[:2]
        windows = np.array([0,0,H,W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x: boxes_clip(x,windows),self.batch_size)
        normalized_boxes = cliped_boxes / np.arange([H,W,H,W])

        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes,scores,self.proposal_count,self.nms_thresh)
            box = tf.gather(normalized_boxes,idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(box)[0],0)
            box = tf.pad(box,[(0,pad_num),(0,0)])# 填充0
            return box
        
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
def trim_zeros_graph(boxes,name=None):
    none_zero = tf.cast(tf.reduce_sum(tf.abs(boxes),axis=1),tf.bool)
    boxes = tf.boolean_mask(boxes,none_zero,name=name)
    return boxes,none_zero

def overlaps_graph(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1,1),[1,1,tf.shape(boxes2)[0]]),[-1,4])
    b2 = tf.tile(boxes2,[tf.shape(boxes1)[0],1])

    b1_y1,b1_x1,b1_y2,b1_x2 = tf.split(b1,4,axis=1)
    b2_y1,b2_x1,b2_y2,b2_x2 = tf.split(b2,4,axis=1)

    y1 = tf.maximun(b1_y1,b2_y1)
    x1 = tf.maximun(b1_x1,b2_x1)
    y2 = tf.minimun(b1_y2,b2_y2)
    x2 = tf.minimun(b1_x1,b2_x1)

    intersection = tf.maximun((y2-y1),0) * tf.maximun((x2-x1),0)
    union = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) + (b2_y2 - b2_y1)*(b2_x2 - b2_x1) - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps



def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
    #去除非0 部分
    proposals,_ = trim_zeros_graph(proposals,name='trim_proposals')
    gt_bboxes,none_zeros = trim_zeros_graph(gt_bboxes,name='trim_bboxes')
    gt_class_ids = tf.boolean_mask(gt_class_ids,none_zeros)

    #计算每个proposals推荐的框预真值的iou
    #取出最合适的推荐
    #计算其对应的delta
    overlaps = overlaps_graph(proposals, gt_bboxes)
    max_iouArg = tf.reduce_max(overlaps,axis=1)
    




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
                    lambda x,y,z:detection_target_graph(x,y,z,self.config),self.config.bash_size,names)

        return outputs


    def compute_output_shape(self,input_shape):
        pass



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
    ''' train'''
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

    '''todo proposal 测试
    '''






    
    




    
































