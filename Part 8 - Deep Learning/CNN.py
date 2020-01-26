# Convolutional Neural Network

# Part 1 -Building the CNN

# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Intialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),
                             input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling

