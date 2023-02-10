import os

import cv2
import numpy as np
import sys

import tensorflow 
from tensorflow.keras.models import Sequential  # Model type to be used
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

print(tensorflow.__version__)

model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Conv2D(16, (5, 5), input_shape=(50,50,1))) # 16 different 5x5 kernels -- so 16 feature maps
model.add(Activation('relu') )                       # activation
model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

# Convolution Layer 2
model.add(Conv2D(32, (5, 5)))                        # 32 different 5x5 kernels -- so 32 feature maps
model.add(Activation('relu'))                        # activation
model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

model.add(Flatten())                                 # Flatten final output matrix into a vector

# Fully Connected Layer 
model.add(Dense(128))                                # 128 FC nodes
model.add(Activation('relu'))                        # activation

# Fully Connected Layer                        
model.add(Dense(10))                                 # final 10 FC nodes
model.add(Activation('softmax'))                     # softmax activation

# Load model weights
model.load_weights(sys.argv[1])

# Check its architecture
model.summary()

for i in range(10):
  img = cv2.imread("{}/{}.png".format(sys.argv[2], i), cv2.IMREAD_GRAYSCALE).astype(float) / 255
  img = np.expand_dims(img, -1)
  img = np.expand_dims(img, 0)
  pred = model.predict(img)
  print(np.argmax(pred))