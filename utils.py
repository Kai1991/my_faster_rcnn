import numpy as np 

# 获取anchor
def anchor_gen(size_X,size_Y,rpn_stride,scales,ratios):
    # size_X 特征图的宽
    # size_Y 特征图的高
    # rpn_stride 特征提取下采样比例
    # scales anchor 面积
    # ratios anchor 的宽高比

    #返回值 ： 特征图上的每个点的 9个框（min_x,min_y,max_x,max_y）
    scales,ratios = np.meshgrid(scales,ratios)
    scales,ratios = scales.flatten(),ratios.flatten()
    scalesY = scales * np.sqrt(ratios)
    scalesX = scales * np.sqrt(ratios)


    shiftX = np.arange(0,size_X) * rpn_stride
    shiftY = np.arange(0,size_Y) * rpn_stride
    shiftX ,shiftY = np.meshgrid(shiftX,shiftY) #anchor 框的宽高
    centerX, anchorX = np.meshgrid(shiftX,scalesX)
    centerY, anchorY = np.meshgrid(shiftY,scalesY)

    anchor_center = np.stack([centerY,centerX],axis=2).reshape(-1,2)
    anchor_size = np.stack([anchorY,anchorX],axis=2).reshape(-1,2)

    boxes = np.concatenate([anchor_center - 0.5*anchor_size,anchor_center + 0.5*anchor_size],axis=1)

    return boxes



if __name__ == "__main__":

    size_Y = 16
    size_X = 16
    rpn_stride = 8

    scales = [2,4,8]
    ratios = [0.5,1,2]

    box = anchor_gen(size_X,size_Y,rpn_stride,scales,ratios)

    print(box.shape)
