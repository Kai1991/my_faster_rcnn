class proposal(KE.Layer):
    def call(self, inputs):
        probs = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas*np.reshape(self.config.RPN_BBOX_STD_DEV, (1, 1, 4))
        prenms_num = min(self.anchors.shape[0], 100)
        idxs = tf.nn.top_k(probs, prenms_num).indices

        probs = batch_slice([probs, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        deltas = batch_slice([deltas, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        anchors = batch_slice([idxs], lambda x:tf.gather(self.anchors, x), self.batch_size)
        refined_boxes = batch_slice([anchors, deltas], lambda x,y:anchor_refinement(x,y), self.batch_size)
        H,W = self.config.image_size[:2]
        windows = np.array([0, 0, H, W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x:boxes_clip(x, windows), self.batch_size)
        normalized_boxes = cliped_boxes / np.array([H, W, H, W])
        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes, scores, self.proposal_count, self.nms_thresh)
            box = tf.gather(normalized_boxes, idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(normalized_boxes)[0],0)
            box = tf.pad(box, [(0, pad_num), (0,0)])
            return box
        proposals_ = batch_slice([normalized_boxes, probs], nms, self.batch_size)
        return proposals_
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)
   





