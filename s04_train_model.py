import numpy as np

sd = 7
np.random.seed(sd)
from tensorflow import set_random_seed
set_random_seed(sd)

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import initializers, regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')

npzfile = 'E:/IP/Labels/labels.npz'

dataset =  np.load(npzfile)
x_train = dataset['X_train']
y_train = dataset['Y_train']

x = x_train/255 # normalizing 

num_classes = 10
eps = 1e-6
dropout_rate_layers = 0.2
dropout_rate_dense = 0.0
maxpool_size = 2
kernel_size = 3
stride = 1
learning_rate = 0.0005
lr_decay = 0.0
bias = False

# CNN MODEL STARTS!

def CNN_model():
    
    model = Sequential()
    
    #layer 1
    model.add(Conv2D(128, kernel_size=kernel_size, 
                strides=stride,
                padding='valid',
                kernel_initializer=initializers.he_normal(seed=sd),
                data_format="channels_last", #(batch, height, width, channels)
                kernel_regularizer=regularizers.l2(0.01), #or kernel_regularizer=None.
                use_bias=bias,
                input_shape=(256,256,3)))
    model.add(BatchNormalization(epsilon=eps, axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout_rate_layers, seed=sd))
    model.add(MaxPooling2D(pool_size=maxpool_size,
                         strides=maxpool_size,
                         data_format="channels_last"))
    
    #layer 2
    model.add(Conv2D(64,kernel_size=kernel_size,
                strides=stride,
                padding='valid',
                kernel_initializer=initializers.he_normal(seed=sd),
                data_format="channels_last", #(batch, height, width, channels)
                kernel_regularizer=regularizers.l2(0.01), #or kernel_regularizer=None.
                use_bias=bias 
               ))
    model.add(BatchNormalization(epsilon=eps, axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout_rate_layers, seed=sd))
    model.add(MaxPooling2D(pool_size=maxpool_size,
                         # strides=maxpool_size,
                         data_format="channels_last"))
    
    
    #layer 3
    model.add(Conv2D(32,kernel_size=kernel_size, 
                #strides=stride,
                padding='valid',
                kernel_initializer=initializers.he_normal(seed=sd),
                data_format="channels_last", #(batch, height, width, channels)
                kernel_regularizer=regularizers.l2(0.01), #or kernel_regularizer=None.
                use_bias=None ))
    model.add(BatchNormalization(epsilon=eps, axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout_rate_layers, seed=sd))
    model.add(MaxPooling2D(pool_size=maxpool_size,
                         # strides=maxpool_size,
                         data_format="channels_last"))
    
    model.add(Flatten())
    
    
    # now the fully connected layers!
    
    model.add(Dense(128, activation='relu',
                   use_bias=bias,
                   kernel_initializer=initializers.he_normal(seed=sd),
                   kernel_regularizer=regularizers.l2(0.01)
                  ))
    model.add(Dropout(rate=dropout_rate_dense, seed=sd))
    model.add(Dense(num_classes, activation='softmax'))
    
    adam = Adam(lr=learning_rate,decay=lr_decay)
    model.compile(loss= 'categorical_crossentropy' , optimizer= adam , metrics=[ 'accuracy' ])
    

    return model

# CNN MODEL ENDS!

model = CNN_model()

model.summary()
# check will store the best model weight which will monitor the validation accuracy not train accuracy
check  = ModelCheckpoint('best.hdf5', monitor = 'val_categorical_accuracy' )
checkpoints = [check]

model.fit(x, y_train, validation_split = 0.3,epochs=15, batch_size=30,verbose=2, callbacks = checkpoints)