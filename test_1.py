class proposal(KE.Layer):
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