import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from keras.models import *
from keras.layers import Flatten, Dense, Lambda,Cropping2D
from keras.layers import *
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Model
%matplotlib inline

##read csv file
lines =[]
with open('/Users/user/Downloads/Self_Driving_Doc/p3_data/data_ori/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
##get image current path / data from csv
images = []
measurements = []
resizeX = 32
resizeY = 16

for i in range(1, len(lines)):
    for j in range(3):
        source_path = lines[i][j]
        filename = source_path.split('/')[-1]
        current_path = '/Users/user/Downloads/Self_Driving_Doc/p3_data/data_ori/IMG/' + filename

        image = cv2.imread(current_path)
        resized = cv2.resize((cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[:,:,1],(resizeX,resizeY))
        images.append(resized)
        
        measurement = float(lines[i][3])
        if j == 1:
            measurement += 0.3
        if j == 2:
            measurement -= 0.3
            
        measurements.append(measurement)

## flip 
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

X_train, y_train = shuffle(X_train, y_train)
X_train = X_train.reshape(X_train.shape[0], resizeY, resizeX, 1)

model = Sequential()#build model
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(16,32,1)))#preprocess
#model.add(Cropping2D(cropping=((70,25),(0,0))))

#model architeture - Lenet
#model.add(Conv2D(2,3,3,subsample=(2,2),activation="relu"))
model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(resizeX,resizeY,1), activation='relu'))
model.add(MaxPooling2D((4,4),(4,4),'valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))

##train
model.compile(loss='mse', optimizer='adam')

batchSize = 128
epoch = 10
model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=epoch, verbose=1)


## save model
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
print("Save Model to Disc")
