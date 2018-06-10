import os
import keras.backend as K
os.environ['KERAS_BACKEND'] = 'theano'
reload(K)
K.set_image_dim_ordering('th')
from PIL import Image as pil_image
from convnetskeras. convnets import convnet
from keras import *
from theano import tensor as T
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda, Conv2D, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import convnetskeras.customlayers
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
splittensor, Softmax4D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
K.set_image_data_format('channels_first')
import cv2



def mean_subtract(img):
    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

def get_alexnet(input_shape,nb_classes,mean_flag): 
	# code adapted from https://github.com/heuritech/convnets-keras

	inputs = Input(shape=input_shape)

	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
		                   name='conv_1', init='he_normal')(mean_subtraction)
	else:
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
		                   name='conv_1', init='he_normal')(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
	    Convolution2D(128,5,5,activation="relu",init='he_normal', name='conv_2_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_2)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3',init='he_normal')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
	    Convolution2D(192,3,3,activation="relu", init='he_normal', name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
	    Convolution2D(128,3,3,activation="relu",init='he_normal', name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

	dense_1 = Flatten(name="flatten")(dense_1)
	dense_1 = Dense(4096, activation='relu',name='dense_1',init='he_normal')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense_2',init='he_normal')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(nb_classes,name='dense_3_new',init='he_normal')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	alexnet = Model(input=inputs, output=prediction)
    
	return alexnet


batch_size = 2
input_size = (3,512,512)
nb_classes = 2
mean_flag = True

alexnet = get_alexnet(input_size,nb_classes,mean_flag)


train_datagen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
 
test_datagen = ImageDataGenerator()                                  
 
train_generator = train_datagen.flow_from_directory(
        './data/train',
        batch_size=batch_size,
        shuffle=True,
        target_size=input_size[1:],
        class_mode='categorical')  
 
validation_generator = test_datagen.flow_from_directory(
        './data/validation',
        batch_size=batch_size,
        target_size=input_size[1:],
        shuffle=True,
        class_mode='categorical')



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
alexnet.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])
 
alexnet.fit_generator(train_generator,
                        samples_per_epoch=1600,
                        validation_data=validation_generator,
                        nb_val_samples=400,
                        nb_epoch=80,
                        verbose=1)


alexnet.save_weights('receipts_512')
