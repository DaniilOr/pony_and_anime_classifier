import cv2
import numpy as np
import os
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
data_train='path to train images'
data_test ='path to test images'
def one_hot_label(img):
	label=img.split('.')[0]
	if ('a' in label):
		ohl=np.array([1,0])
	else:
		ohl=np.array([0,1])
	return ohl
def train_data_with_label():
	train_images=[]
	for i in tqdm(os.listdir(data_train)):
		path=os.path.join(data,i)
		img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		train_images.append([np.array(img),one_hot_label(i)])
	shuffle(train_images)
	return train_images

def test_data_with_label():
	test_images=[]
	for i in tqdm(os.listdir(data_test)):
		path=os.path.join(data,i)
		img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		test_images.append([np.array(img),one_hot_label(i)])
	return test_images

	

						
#MAIN BODY
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
training_images = train_data_with_label()
testing_images = test_data_with_label()
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_images])
#I had some problevs with the standart optimizer
import theano
theano.config.optimizer="merge"							
							
#The model. You may change activation function for better accuracy.
model=Sequential()
model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=32, kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))	

model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu',input_shape=[64,64,1]))
model.add(Dropout(rate=0.5))
model.add(Dense(2,activation='softmax'))
optimizer=Adam(lr=1e-3)							


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])		
ep = "Number of epochs, you need"
b_s = "Batch size, you want"
model.fit(x=tr_img_data, y=tr_lbl_data,epochs=ep,batch_size=b_s)
