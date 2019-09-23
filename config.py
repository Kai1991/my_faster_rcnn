import numpy as np

class Config():
    BACKBONE = 'resnet50'  #  'resnet50' 'resnet101' 使用的backbone
    TRAIN_BN = True # 是否训练
    TOP_DOWN_PYRAMID_SIZE = 256 # 特征金字塔最上层大小
    featureMap_size = [8, 8]
    scales = [4, 8, 16]
    ratios = [0.5, 1, 2]
    rpn_stride = 8
    anchor_stride = 1
    train_rois_num = 100
    image_size = [64, 64]
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    num_before_nms = 300
    max_gt_obj = 30
    num_proposals_train = 21
    num_proposals_ratio = 0.333
    batch_size = 20