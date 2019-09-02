from keras.models import Model 
import keras.layers as KL
import keras.backend as K
import keras.engine as KE
import tensorflow as tf 
import numpy as np 


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


##########################################################################
#
#                                                                            Resnet
#
##########################################################################

def building_block(filters,block):
    if block != 0:
        stride = 1
    else:
        stride = 2
    def f(x):
        y = KL.Conv2D(filters,(1,1),strides=stride)(x)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(filters,(3,3),padding='same')(x)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(4 * filters,(1,1))(x)
        y = KL.BatchNormalization(axis=3)(y)
        
        if block == 0:
            shortcut = KL.Conv2D(4 * filters,(1,1),strides=stride)(x)
            shortcut = KL.BatchNormalization(axis=3)(shortcut)
        else:
            shortcut = x
        
        y = KL.add()([y,shortcut])
        y = KL.Activation('relu')(y)
        return y
    return f


def resnet_feature_extractor(inputs):
    
































