/*
* <This code is a part of the NA62 collaboration>
* Copyright (C) <2019-2023>  <APE group INFN roma + Andrea>
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns                   #Plot ROC
from itertools import cycle             #Plot ROC


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D,MaxPool2D, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from tensorflow import keras  # load_model
from tensorflow.keras.utils import to_categorical # used by get data to transform unidimensional target in a 4 dim  vector (one dimension per class)

from sklearn.model_selection import train_test_split # used in data preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc      # Compute ROC


from apesettings import FOLDER
from apesettings import CreateDir

FOLDER = "data/"
#########################
# CONSTANTS (module wide)
#########################
#batch_size = 1024
batch_size = 16384
N_LABEL=4  # number of output classes
MAXLABEL=4

##################
# CONFUSION MATRIX
##################
class ConfusionMatrix:

	def __init__(self):
		# confusion matrix is an array of integer mumbners
		self.cm   = np.zeros(shape=MAXLABEL*MAXLABEL, dtype=int)
		# normalization
		self.sumA  = 0
		self.sumH = np.zeros(shape=MAXLABEL          , dtype=int)
		self.sumV = np.zeros(shape=MAXLABEL          , dtype=int)
		# performance variables
		self.effici = np.zeros(shape=MAXLABEL          , dtype=float)
		self.purity = np.zeros(shape=MAXLABEL          , dtype=float)
		self.under  = np.zeros(shape=MAXLABEL          , dtype=float)
		self.over   = np.zeros(shape=MAXLABEL          , dtype=float)

	def getIndex(self,row,col):
		return col + row*MAXLABEL


	def calculateNormalization(self):
		# calculate total, horizontal and vertical normalization
		for r in range(MAXLABEL):
			for c in range(MAXLABEL):
				idx = self.getIndex(r,c)
				count = self.cm[idx]
				self.sumA    += count
				self.sumH[r] += count
				self.sumV[c] += count

	def calculatePerformance(self):
		# calculate effinciency and purity per class using normalization factors
		for c in range(4):
			for r in range(4):
				idx = c+MAXLABEL*r
				count = self.cm[idx]
				if (r == c):
					self.effici[r]  = count/self.sumH[r]
					self.purity[c]	= count/self.sumV[c]
				else:
					if (r < c):
						self.under[c] += count/self.sumV[c]
					if (r > c):
						self.over[c] += count/self.sumV[c]

		# in percentage
		for j in range(4):
			self.effici[j]  = 100* self.effici[j]
			self.purity[j]  = 100* self.purity[j]
			self.over  [j]  = 100* self.over  [j]
			self.under [j]  = 100* self.under [j]



	def read(self, infile):
		# read file as an array of string
		fin = open(infile, "r")
		cm = fin.read().split()
		fin.close()
		# cast to private array of intgers
		for r in range(MAXLABEL):
			for c in range(MAXLABEL):
				idx = self.getIndex(r,c)
				self.cm[idx] = int(cm[idx])

		self.calculateNormalization()
		self.calculatePerformance()


	def getTotalEvents(self):
		return self.sumA

	def getTotalEventsClass(self,label):
		return self.sumH[label]

	def getTotalEventsClassifiedAs(self,label):
		return self.sumV[label]

	def getPurity(self,label):
		return self.purity[label]

	def getEfficiency(self,label):
		return self.effici[label]

	def getUnderContamination(self,label):
		return self.under[label]
	def getOverContamination(self,label):
		return self.over[label]


	def dump(self):

		print("Total Events %8d" % self.sumA, )

		for r in range(4):
			print("Total events of class %d is %8d   (%4.2lf %%)" % (r,self.sumH[r],100*self.sumH[r]/self.sumA))
		for c in range(4):
			print("Total events classified as %d is %8d   (%4.2lf %%)" % (c,self.sumV[c],100*self.sumV[c]/self.sumA))


		for j in range(4):
			print("Class %2d " % j , end = '')
			print("Efficiency %4.1lf "          % (self.effici[j]  ), end = '  ')
			print("Purity %4.1lf "              % (self.purity[j]	  ), end = '  ')
			print("OverContamination %4.1lf "   % (self.over[j]        ), end = '  ')
			print("UnderContamination  %4.1lf " % (self.under[j]	  ), end = '  ')
			print()




	def exportHuman(self,outfilename):

		fout = open(outfilename, "w")

		print("Total Events %8d" % self.sumA,file = fout)

		for r in range(4):
			print("Total events of class %d is %8d   (%4.2lf %%)" % (r,self.sumH[r],100*self.sumH[r]/self.sumA),file = fout)
		for c in range(4):
			print("Total events classified as %d is %8d   (%4.2lf %%)" % (c,self.sumV[c],100*self.sumV[c]/self.sumA),file = fout)


		for j in range(4):
			print("Class %2d "                  % j                , end = ' ' , file = fout)
			print("Efficiency %4.1lf "          % (self.effici[j] ), end = ' ' , file = fout)
			print("Purity %4.1lf "              % (self.purity[j] ), end = ' ' , file = fout)
			print("OverContamination %4.1lf "   % (self.over[j]   ), end = ' ' , file = fout)
			print("UnderContamination  %4.1lf " % (self.under[j]  ), end = ' ' , file = fout)
			print("",file = fout)

		fout.close()


	def export(self, outfilename):

		fout = open(outfilename, "w")

		print("%8d" % self.sumA, file = fout, end = ' ')

		for j in range(MAXLABEL):
			print("%8d "    % self.sumH[j]        , file = fout, end = ' ')
			print("%8d "    % self.sumV[j]        , file = fout, end = ' ')
			print("%4.1lf " % self.effici[j]      , file = fout, end = ' ')
			print("%4.1lf " % self.purity[j]      , file = fout, end = ' ')
			print("%4.1lf " % self.under[j]       , file = fout, end = ' ')
			print("%4.1lf " % self.over[j]        , file = fout, end = ' ')

		print("",file = fout)
		fout.close()
		print("Report ready in file %s " % outfilename)







############################
# NEURAL NETWORK CLASSIFIERS
############################
def DenseModel(shape_in):
    inTnsr  = Input(shape_in, name="input1") # default dtype is float32
    h1      = Dense( 64        , name = 'fc1' ) (inTnsr)
    h1a     = Activation("relu", name = 'act1') (h1)
    h2      = Dense( 16        , name = 'fc2' )(h1a)
    h2a     = Activation("relu", name = 'act2') (h2)
    h3      = Dense( 4         , name = 'fc3' )(h2a)
    outTnsr = Activation("softmax", name="softmax")(h3)

    model  = Model(inputs=[inTnsr], outputs=[outTnsr])

    return model

    #AL DSP 5%, LUT 5% VERY GOOD
def ConvModel(shape_in):
    inTnsr  = Input     (shape_in, name="input1") # default dtype is float32
    x       = Conv2D    (filters   = 8, kernel_size = 3, padding ='same', name = 'conv1') (inTnsr)
    x       = Activation("relu"                                         , name = 'act1' ) (x)
    x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp1') (x)
    x       = Conv2D    (filters   = 8, kernel_size = 3, padding ='same', name = 'conv2') (x)
    x       = Activation("relu"                                         , name = 'act2' ) (x)
    x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp2') (x)
    x       = Flatten()(x)
    x       = Dense( 16           , name = 'fc3'  ) (x)
    x       = Activation("relu"   , name = 'act3' ) (x)
    x       = Dense(4             , name = 'fc4'  ) (x)
    outTnsr = Activation("softmax", name="softmax") (x)
    model  = Model(inputs=[inTnsr], outputs=[outTnsr])

#AL adapted version of original version by Ciardiel
# def ConvModel(shape_in):
#     inTnsr  = Input(shape_in, name="input1") # default dtype is float32
#     x = BatchNormalization()(inTnsr)
#     x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#     x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#     x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
#     x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#     x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#     x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#     x = Conv2D (filters =32, kernel_size =3, padding ='same', activation='relu')(x)
#     x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
#     x = Flatten()(x)
#     x = Dense(units = 64, activation ='relu')(x)
#     x = Dense(64, name ='fc1_extra3')(x)
#     x = Activation("elu", name='act1_extra3') (x)
#     x = BatchNormalization()(x)
#     x = Dense(64, name ='fc2e')(x)
#     x = Activation("relu", name='act2e') (x)
#     x = Dropout(0.5)(x)
#     out= Dense(4, name ='fc3')(x)
#     outTnsr = Activation("softmax", name="softmax")(out)
#     model  = Model(inputs=[inTnsr], outputs=[outTnsr])

# def ConvModel(shape_in):

#     inTnsr  = Input     (shape_in, name="input1") # default dtype is float32
#     x       = Conv2D    (filters   = 16, kernel_size = 4, padding ='same', name = 'conv1') (inTnsr)
#     x       = Activation("relu"                                         , name = 'act1' ) (x)
#     x       = Conv2D    (filters   = 16, kernel_size = 4, stride = 2, padding ='same', name = 'conv2') (x)
#     x       = Activation("relu"                                         , name = 'act2' ) (x)
# #    x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp1') (x)
#     x       = Conv2D    (filters   = 16, kernel_size = 4, padding ='same', name = 'conv3') (x)
#     x       = Activation("relu"                                         , name = 'act3' ) (x)
#     x       = Conv2D    (filters   = 16, kernel_size = 4, stride = 2, padding ='same', name = 'conv4') (x)
#     x       = Activation("relu"                                         , name = 'act4' ) (x)
# #    x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp2') (x)
#     x       = Conv2D    (filters   = 16, kernel_size = 2, padding ='valid', name = 'conv5') (x)
#     x       = Activation("relu"                                           , name = 'act5' ) (x)
#     x       = Conv2D    (filters   = 16, kernel_size = 4, stride = 2, padding ='same', name = 'conv6') (x)
#     x       = Activation("relu"                                         , name = 'act6' ) (x)
# #    x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='valid' , name = 'maxp3') (x)
#     x       = Flatten()(x)
#     x       = Dense( 32           , name = 'fc1'  ) (x)
#     x       = Dropout( 0.5,  name = 'dp1'  ) (x)
#     x       = Activation("relu"   , name = 'act7' ) (x)
#     x       = Dense( 16           , name = 'fc2'  ) (x)

#     x       = Dropout( 0.5,  name = 'dp2'  ) (x)
#     x       = Activation("relu"   , name = 'act8' ) (x)
#     x       = Dense(4             , name = 'fc3'  ) (x)
#     outTnsr = Activation("softmax", name="softmax") (x)
#     model  = Model(inputs=[inTnsr], outputs=[outTnsr])



    #AL good perf but 18% of DSPs
    # inTnsr  = Input     (shape_in, name="input1") # default dtype is float32
    # x       = Conv2D    (filters   = 8, kernel_size = 3, padding ='same', name = 'conv1') (inTnsr)
    # x       = Activation("relu"                                         , name = 'act1' ) (x)
    # x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp1') (x)
    # x       = Conv2D    (filters   = 8, kernel_size = 3, padding ='same', name = 'conv2') (x)
    # x       = Activation("relu"                                         , name = 'act2' ) (x)
    # x       = MaxPool2D (pool_size = 2, strides     = 2, padding ='same', name = 'maxp2') (x)
    # x       = Flatten()(x)
    # x       = Dense( 32           , name = 'fc3'  ) (x)
    # x       = Dropout( 0.5,  name = 'dp1'  ) (x)
    # x       = Activation("relu"   , name = 'act3' ) (x)
    # x       = Dense(4             , name = 'fc4'  ) (x)
    # outTnsr = Activation("softmax", name="softmax") (x)
    # model  = Model(inputs=[inTnsr], outputs=[outTnsr])


#AL original version by Ciardiel
#    inTnsr  = Input(shape_in, name="input1") # default dtype is float32
#    x = AveragePooling2D(pool_size=(3, 3),strides=2, padding='same')(inTnsr)
#    x = AveragePooling2D(pool_size=(3, 3),strides=2, padding='same',name="input_compressed")(x)
#    x = AveragePooling2D(pool_size=(3, 3),strides=2, padding='same',name="input_compressed_2")(x)
#    x = BatchNormalization()(x)
#    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
#    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
#    x = Conv2D (filters =32, kernel_size =3, padding ='same', activation='relu')(x)
#    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
#    x = Flatten()(x)
#    x = Dense(units = 64, activation ='relu')(x)
#    x = Dense(64, name ='fc1_extra3')(x)
#    x = Activation("elu", name='act1_extra3') (x)
#    x = BatchNormalization()(x)
#    x = Dense(64, name ='fc2e')(x)
#    x = Activation("relu", name='act2e') (x)
#    x = Dropout(0.5)(x)
#    out= Dense(4, name ='fc3')(x)
#    outTnsr = Activation("softmax", name="softmax")(out)
#    model  = Model(inputs=[inTnsr], outputs=[outTnsr])



    return model



def Predict(model,x_test):
        y_pred = model.predict(x_test)
        print("y_pred shape: ", y_pred.shape)
        return y_pred

def Train(model,x,y, epoch):
	split = 0.1
	my_callbacks = [
		tf.keras.callbacks.EarlyStopping(patience=2),
		tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
		tf.keras.callbacks.TensorBoard(log_dir='./logs'),
	]


	classes = [[1., 0., 0., 0.],
		[0., 1., 0., 0.],
 		[0., 0., 1., 0.],
 		[0., 0., 0., 1.]]


#	class_weight = {0: 0.0001, 1: 0.01,2: 1.0,3: 100.0} # settings for single run
#	class_weight = {0: 0.0001, 1: 0.01,2: 1.0,3: 100.0} # settings for single run

	yClass = np.argmax(y,axis=1)

	print(yClass[0:100])



	from sklearn.utils.class_weight import compute_class_weight



	class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = [0,1,2,3],
                                        y = yClass                                                    
                                      )
	class_weights = dict(zip([0,1,2,3], class_weights))

	print(class_weights)


	

	#story = model.fit(x,y, verbose=2, batch_size=batch_size, epochs=epoch, validation_split=split, callbacks=my_callbacks)
#	story = model.fit(x,y, verbose=2, batch_size=batch_size, epochs=epoch, validation_split=split)
	story = model.fit(x,y, verbose=2, batch_size=batch_size, epochs=epoch, validation_split=split, class_weight = class_weights)
	#story = model.fit(x,y, verbose=1, batch_size=batch_size, epochs=epoch, validation_split=split)

	#print(history)
	#print(history.params)
	#print(history.history.keys())
	#print(history.history['loss'])

	return story


def SaveModel(model,name):
	folder = FOLDER + 'result/model/'
	my_model = folder + name
	print('Saving model in %s' % my_model)
	model.save(my_model)


def LoadModel(name):

	folder = FOLDER + 'result/model/' + name + '/' +name
	print('Loading model %s' % folder)
	model = keras.models.load_model(folder)
	print(model.summary())
	return model

###########################
# Data IO and Preprocessing
###########################


def get_x_test(dataset):

        print('Loading test features from dataset %s' % dataset)
        x_test = np.load( FOLDER + 'result/data/' + 'x_test_{}.npy'.format(dataset))
        print("x_test shape: ", x_test.shape)
        return x_test

def get_x_train(dataset):

        print('Loading training features from dataset %s' % dataset)
        x_train = np.load( FOLDER + 'result/data/' + 'x_train_{}.npy'.format(dataset))
        print("x_train shape: ", a.shape)
        return x_train

def get_y_test(dataset):

        print('Loading test predictions from dataset %s' % dataset)
        y_test = np.load( FOLDER + 'result/data/' + 'y_test_{}.npy'.format(dataset))
        print("y_test shape: ", y_test.shape)
        return y_test

def get_y_train(dataset):

        print('Loading training predictions from dataset %s' % dataset)
        y_train = np.load( FOLDER + 'result/data/' + 'y_train_{}.npy'.format(dataset))
        print("y_train shape: ", y_train.shape)
        return y_train

def transform1d(a):
	print("shape in : ", a.shape)
	b = np.argmax(a, axis=1) # transform actual output (4d) in a 1d  array using max value
	print("shape out: ", b.shape)
	return b



def get_vectors_random():
        # create test vectors for features and labels
        n_features  =  64
        n_labels    =   4
        n_samples   = 100
        test_input  = np.random.random((n_samples, n_features))
        test_target = np.random.random((n_samples, n_labels  ))
        return test_input, test_target








def getData(dataset = "2450206",label = "np_track"):

	# load data from a json file
	name = 'data/' + 'data_{}.json'.format(dataset)
	print('Loading %s' % name)
	with open(name, 'r') as f:
        	data = json.load(f)

	nlist  = len(data)  # number of lists in the dictionary
	nevent = len(data['hitlist']) # number of events
	# NOTE that lists have the same number of events by construction, see ape62.read_data...
	# in case you want check
	if(1):
		for key in data:
			print("%6d items in list %s" % ( len(data[key]), key ) )


	# Important! label selection here
	name = label


	###############################
	# LABELs processing
	################################
	# goal is to have a vector ready for training
	print('Processing selected label (%s)' % name )
	arr  = np.array(data[name]) 		# get the list. Labels have 1 integer for each event
	arr[arr>3]=3 				# form the 3+ class with events that have label equal or greater than two (RingDumper accept events with moltiplicity from 0 to 7)
	y = to_categorical(arr, N_LABEL) 	# label is trasformed in a 4-Dimension (one per output class)


	######################
	# FEATURES  processing
	######################
	# this is going to be the input of the Neural Network
	# the hitlist  is transformed accordingly to the model
	# current options are:
	# 	model dense   64 input hitlist normalized
	#	 model dense 2048 input positional encoding (a 1 is placed in a vector of 2048 in correspodance with the hit)
	#	 model conv  NxN  input is an image of N x N pixel, this is a squared resampling of the pmt array after the mirror corrections (a resampling of the event Display)
	# NOTE: a cut on the number of hits (hit<=64) is operated while reading data with ape62.read...
	arr = np.zeros((nevent,64)) 		# fixed size array # print(arr.shape)
	for i in range(nevent):
		list = data['hitlist'][i] 	# hitlist of the i-th event
		l = (len(list)) 		# number of hits
		arr[i][:l] = list[:l]


	# TODO
	# Print some event on plain text
	# Draw some event on event display
	# Draw dome event in standard format
	# Draw something else

	######################
	# Features processing
	######################
	# finally the feature array can be processed to obtain the input vector for the Neural Network
	# TODO: add a switch and other pre-processing method i.e. OHE and IMAGE
	x = arr		# as it is



#	print('Feature preprocessing  = Normalization by 2048')
#	x = arr/2048



	#to check data
	print('Example data')
	if(1):
		for i in range(1):
			print('*************')
			print('Event %6d' % i)
			print('*************')
			for key in data:
				print("%10s:" %  key, end = ' ')
				print(data[key][i])
	#		print('List:')
#			print(list)
#			print('List Len')
#			print(len(list))
			print('Array')
			print(arr[i])
			print('Feature preprocessed')
			print(x[i])


	if(1):
        	print(type(x))
        	print(x.shape)
        	print(type(y))
        	print(y.shape)

	return x,y




def SaveAll(x,y,dataset):
        folder = 'results/preprocess/'
        CreateDir(folder)
        np.save(folder + 'x_all_{}.npy'.format(dataset) , x)
        np.save(folder + 'y_all_{}.npy'.format(dataset) , y)
        print('Validation: X=%s, y=%s' % (x.shape, y.shape))
        print('Data saved in %s' % folder)
        return


def ShuffleSplit(x,y,dataset):
        ###  shuffle and split in train and validation set
	# save all variables to disk
	# return variables for training
	test_size    = 0.1		# 0.1 means 10% saved for testing
	random_state = 0  		# seed for riproducibility in random generator

	x_train, x_test , y_train, y_test  = train_test_split (x,y, test_size=test_size , random_state = random_state  )

	print('Data shuffled (random_state = %d) ' % (random_state))
	print('Data splitted in train (%.2lf), test (%.2lf)' % (1-test_size,test_size))

	folder = FOLDER + 'result/preprocess/'

	CreateDir(folder)

	#name = folder + 'x_train_{}.npy'.format(dataset)

	np.save(folder + 'x_train_{}.npy'.format(dataset) , x_train)
	np.save(folder + 'x_test_{}.npy'.format(dataset)  , x_test)
	np.save(folder + 'y_train_{}.npy'.format(dataset) , y_train)
	np.save(folder + 'y_test_{}.npy'.format(dataset)  , y_test)

	print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
	print('Test : X=%s, y=%s' % (x_test.shape , y_test.shape))

	print('Data saved in %s' % folder)

	return


########################
# PERFORMANCE EVALUATION
########################

def ROC(y_test,y_pred, ofolder):
	fpr,tpr,roc_auc = Compute_ROC(y_test, y_pred)
	outPdf = ofolder + 'roc.pdf'
	Plot_ROC(fpr,tpr,roc_auc,outPdf)


def Compute_ROC(y_test, y_pred):

        # Compute ROC curve and ROC area for each class
        fpr = {} #equivalent to dict()
        tpr = {}
        roc_auc = {}
        for i in range(N_LABEL):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        return fpr,tpr,roc_auc

def Plot_ROC(fpr,tpr,roc_auc, outPdf):
        palette = sns.color_palette("bright") # Save a palette to a variable
        sns.palplot(palette) # Use palplot and pass in the variable
        lw = 2
        plt.figure()

        colors = cycle(['green', 'orange', 'brown','blue'])
        for i, color in zip(range(N_LABEL), colors):
                #plt.plot(fpr[i], tpr[i], color=color, linewidth=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
                plt.plot(tpr[i], fpr[i], color=color, linewidth=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', linewidth=lw)

#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')

        plt.xlabel('Signal Efficiency (True Positive Rate)')
        plt.ylabel('Background Efficiency (False Positive Rate)')
        plt.title('Multi-Class ROC Curve')
        plt.legend(loc="upper left")
        #prefix = FOLDER + 'result/plot/' + model_name
        #linear
#        plt.ylim([0.0, 1.05])
#        plt.xlim([0.0, 1.05])
#        name = prefix +'_roc_linear.pdf'
#        plt.savefig(name, bbox_inches='tight', dpi=200)
#        print('File %s ready' % name)

        #semilog
        plt.yscale('log')
        plt.ylim([1e-3, 1.00])
        plt.xlim([0.0, 1.0])
#        name = prefix +'_roc_semilog.pdf'
#        plt.savefig(name, bbox_inches='tight', dpi=200)
#        print('File %s ready' % name)

        plt.savefig(outPdf, bbox_inches='tight', dpi=200)
        print('File %s ready' % outPdf)



def ClassificationReport(y_true, y_pred, ofolder,figtitle='ConfusionMatrix'):


	y_pred = transform1d(y_pred)
	y_true = transform1d(y_true)

	outPdf = ofolder + 'cm.pdf'
	outTxt = ofolder + 'cm.txt'

	# CM = confusion matrix
	cm    = []  	# list of matrices, each one has different normalization
	dis   = []	# relative display object for plotting
	opt   = ['all','true','pred'] # possible normalization are 3

	# here four matrices are calculated, the first reports counts
	# the other three have different normalizations (from scikitlearn)

	for i in range(4):
		if i==0:
			m = confusion_matrix (y_true, y_pred)
		else:
			m = confusion_matrix (y_true, y_pred, normalize = opt[i-1])

		cm.append(m)
		d = ConfusionMatrixDisplay(m)
		dis.append(d)

	# Figure
	fig, axs = plt.subplots(1, 4, figsize=(32, 8))

	title = ['Counts',
		'Counts Normalized',
		'Counts Normalized per true Label (horizontal)',
		'Counts Normalized per prediction (vertical)']

	for i, ax in enumerate(fig.axes):
		ax.set_title(title[i])
		format = '.2f'
		if i==0:
			format='9d'
		#else:
			#ax.set_zlim(0, 1) # I could not find a way to fix the z axis range!
			#plt.zlim(0.0,1.0)
			#ax.set_zlim(0.0,1.0)
			#plt.zlim([0.0, 1.0])
		dis[i].plot(ax=ax,cmap='Blues' ,values_format=format)
	fig.suptitle(figtitle, fontsize=16)
	fig.savefig(outPdf)
	print('Confusion matrix plot ready in %s' % outPdf)



	# export confusion matrix (one-line version)
	filename = ofolder + "matrix.txt"

	fout = open(filename, "w")
	for r in range(4):
		for c in range(4):
			print("%8d" % cm[0][r][c], file = fout, end = ' ')
	print(" ", file = fout)
	fout.close()
	print("Confusion matrix saved to %s " % filename)


	# declare a ConfusionMatrix object and use it to produce human Ëšfriendly and computer friendly reports
	# it ingest the previously exported counting matrix (which was saved to file 'filename')
	matrix = ConfusionMatrix()
	matrix.read(filename)
	#matrix.dump()
	matrix.exportHuman(ofolder + "report_human.txt")
	matrix.export(ofolder + "report.txt")


	# Report (complete with all ML enthusiast stuff but never used in practice in this project)
	cm0 = cm[0] # we grab just the 'counting' comfusion matrix to do all the job
	cm1 = cm[1] # the others are for printing in the report
	cm2 = cm[2]
	cm3 = cm[3]
	target_names = ['0  rings', '1  ring ','2  rings', '3+ rings']
	report = classification_report(y_true, y_pred,target_names = target_names, digits= 3)
	fout = open(outTxt, "w")
	print(
		f" Confusion Matrix\n"
		f"{cm0}\n\n\n"
		f" Normalized All\n"
		f"{cm1}\n\n\n"
		f" Normalized per rows (Efficiency)\n"
		f"{cm2}\n\n\n"
		f" Normalized per columns (Purity)\n"
		f"{cm3}\n\n\n"

		f"Classification report:\n"
		f"{report}\n",
		file = fout)
	fout.close()

	# Report Extra
	# True and False rates are calculated for all the classes
	# and combined to form other kind of metrics
	# everything is appended to the output plain text file
	FP = cm0.sum(axis=0) - np.diag(cm0)
	FN = cm0.sum(axis=1) - np.diag(cm0)
	TP = np.diag(cm0)
	TN = cm0.sum() - (FP + FN + TP)

	TPR = TP/(TP+FN) # Sensitivity, hit rate, recall, or true positive rate
	TNR = TN/(TN+FP) # Specificity or true negative rate
	PPV = TP/(TP+FP) # Precision or positive predictive value
	NPV = TN/(TN+FN) # Negative predictive value
	FPR = FP/(FP+TN) # Fall out or false positive rate
	FNR = FN/(TP+FN) # False negative rate
	FDR = FP/(TP+FP) # False discovery rate
	ACC = (TP+TN)/(TP+FP+FN+TN) # Overall accuracy

	int_formatter = "{:6d}".format
	np.set_printoptions(formatter={'int_kind':int_formatter})

	fout = open(outTxt, "a")
	print('Number of FP,FN,TP,TN', file = fout)
	print( FP, file = fout)
	print( FN, file = fout)
	print( TP, file = fout)
	print( TN, file = fout)

	float_formatter = "{:6.2f}".format
	np.set_printoptions(formatter={'float_kind':float_formatter})

	print('True Positive Rate (Sensitivity) ', file = fout)
	print(TPR, file = fout)
	print('True Negative  Rate (Specificity)', file = fout)
	print(TNR, file = fout)
	print('Precision or positive predictive value', file = fout)
	print(PPV, file = fout)
	print('Negative predictive value', file = fout)
	print(NPV, file = fout)
	print('Fall out or false positive rate', file = fout)
	print(FPR, file = fout)
	print('False negative rate', file = fout)
	print(FNR, file = fout)
	print('False discovery rate', file = fout)
	print(FDR, file = fout)
	print('Overall Accuracy', file = fout)
	print(ACC, file = fout)
	fout.close()

	print("Report ready in file %s " % outTxt)


###############
# PLOTTING
###############
def PlotLoss(x,outPdf):
	# input: a training_history (the output of model.fit)
	plt.plot(x.history['loss']    , label='training data')
	plt.plot(x.history['val_loss'], label='validation data')
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(loc="upper right")
	#plt.ylim((0.0,3.0))
	plt.savefig(outPdf, bbox_inches='tight', dpi=200)
	print('Training Loss ready at %s' % outPdf)
	plt.close()

def PlotAccuracy(x,outPdf):
	# input: a training_history (the output of model.fit)
	plt.plot(x.history['accuracy']    , label='training data')
	plt.plot(x.history['val_accuracy'], label='validation data')
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="upper right")
	#plt.figure(figsize=(9,9))
	#plt.ylim((0.0,3.0))
	plt.savefig(outPdf, bbox_inches='tight', dpi=200)
	print('Training Accuracy ready at %s' % outPdf)
	plt.close()



def EventPlot(dataset): # should be moved to ape62 module
	print('%s to be implemented' % EventPlot.__name__)
        ## plot on pdf few events
        ## retrieve data from disk
        #data = ...
        #from matplotlib import pyplot as plt
        #for i in range(9):
        #        plt.subplot(330 +1 +i)  #define subplot
        #        plt.plot(data[i]) # plot raw data
        ##save the figure
        #outpdf = FOLDER + "result/plot/plot_raw_events.pdf"
        #plt.savefig(outpdf, bbox_inches='tight', dpi=200)

def EventPrint(dataset): # to be moved to ape62odule
	print('%s to be implemented' % EventPrint.__name__)
        ## print on file few events
        ## retrieve data from disk
        #data = ...





################################################
# Following method are still work in progress on 2022 March 3rd
#################################################
# they deal with hls_model performance calculation and plotting
# to be compared with keras evaluation and plottinf above
# goal would be methods for individual model and class evaluation
#  and also methods to compare different models




import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score


import pandas as pd # rocData as done by HLS4ML

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from apeml import LoadModel
from apeml import get_x_test
from apeml import get_y_test
from apeml import transform1d
from apeml import Predict
from apesettings import FOLDER
#import hls4ml


def rocCurveMulticlass(y_true,y_pred,name):
        labels = ['zero','one','two','three+']
        n_class = 4

        fpr = {}
        tpr = {}
        thresh = {}

        # y_test and y_pred are matrices with nevents rows and  4 numbers columns
        # roc_curve wants 1d array (one number per event)
        # we loop on i considering a different target prediction at each iteration of the loop

        linestyle = ['solid', 'dashed', 'dashdot','dotted']
        color=['orange','green','blue','red']
        linename= ['Class 0 vs Rest','Class 1 vs Rest','Class 2 vs Rest','Class 3+ vs Rest']

        df = pd.DataFrame()
        for i, label in enumerate(labels):
                print(i,end = ' ')
                print(label)
                df[label + '_true'] = y_true[:,i]
                df[label + '_pred'] = y_pred[:,i]
                fpr[label], tpr[label], thresh[label] = roc_curve(df[label+'_true'],df[label+'_pred'])
                plt.plot(fpr[label], tpr[label], linestyle=linestyle[i],color=color[i],label=linename[i])


        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')

        outPdf = FOLDER + 'result/plot/' + name +'_ROC_multiclass.pdf'
        plt.savefig(outPdf, bbox_inches='tight', dpi=300)
        print('%s ready' %outPdf)



def curvePR(y_true,y_pred,name): # precision-recall curves


        y_pred_1d = transform1d(y_pred)
        y_true_1d = transform1d(y_true)



        positive_label = 0  # 0,1,2,3
        # get precision and recall values
        precision, recall, thresholds = precision_recall_curve(y_true_1d, y_pred_1d, pos_label=positive_label)

        # average precision score
        avg_precision = average_precision_score(y_true,y_pred)

        # precision auc
        pr_auc = auc(recall, precision)

        # plot
        plt.figure(dpi=150)
        plt.plot(recall, precision, lw=1, color='blue', label=f'AP={avg_precision:.3f}; AUC={pr_auc:.3f}')
        plt.fill_between(recall, precision, -1, facecolor='lightblue', alpha=0.5)
        plt.title('PR Curve for Multi-label Dense classifier')
        plt.xlabel('Recall (TPR)')
        plt.ylabel('Precision')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()

        outPdf = FOLDER + 'result/plot/' + name +'_PR.pdf'
        plt.savefig(outPdf, bbox_inches='tight', dpi=300)
        print('%s ready' %outPdf)



def rocData(y_true, y_pred, labels):

        # returns dictionaries organized per labels
        fpr = {}
        tpr = {}
        auc1 = {}
        thr = {}


        # Y data, an array of 4d vectors, binary values,matrice: riga = evento, per ogni evento 4 numeri (tre zeri e un 1 i.e. la risposta giusta)
        # EVENT_1 (0,0,1,0)
        # EVENT_2 (0,1,0,0)
        # loop on labels: l' iteratre e' fatto [ [0: zero rings], [1: one ring], 2: two rings....

        # wrap data in a Dataframe to pass them easily to roc_curve method
        df = pd.DataFrame()
        for i, label in enumerate(labels):
                df[label + '_true'] = y_true[:,i]
                df[label + '_pred'] = y_pred[:,i]
                fpr[label], tpr[label], thr[label] = roc_curve(df[label+'_true'],df[label+'_pred'])
                auc1[label] = auc(fpr[label], tpr[label])
        return fpr, tpr, auc1, thr



def plotRoc(fpr, tpr, auc, labels, linestyle, legend=False):

        # following HLS4ML people the ROC has inverted axis and  False Positive Rate (y) in logarithic
        # typically the ROC is given linear  with TPR(y ) vs FPR(x)

        for i, label in enumerate(labels):
                plt.plot(fpr[label],tpr[label],label='%s ring(s), AUC = %.1f%%'%(label.replace('j_',''),auc[label]*100.),linestyle=linestyle)
        plt.semilogy()
        ax = plt.gca()
        ax.set_xlabel('Signal Efficiency (True Positive Rate)', labelpad=24)
        ax.set_ylabel('Background Efficiency (False Positive Rate)', labelpad=24)

        plt.ylim(1e-3,1)
        plt.grid(True)
        if legend: plt.legend(loc='upper right')
        # ADD TEXT
        ##    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)


def plotThrScan(fpr,tpr,thr,labels,name,legend=False):


        for i,label in enumerate(labels):
#               plt.plot(tpr[label],thr[label],label='%s ring(s)' % label)
#               plt.plot(fpr[label],thr[label],label='%s ring(s)' % label)

                a=fpr[label]
                b=tpr[label]
                c=thr[label]
                plt.plot(b[1:],c[1:],label='%s ring(s)' % label) # skip firt element which conventionally contain ymax +1 = 2 so detroy the plot
                plt.plot(a[1:],c[1:],label='%s ring(s)' % label)

                # save data on txt file
                np.savetxt('test_{}.txt'.format(label), (a,b,c), fmt='%6.3f')


        # Draw a point
        x = [0.5]
        y = [0.5]
        plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red")

        ax = plt.gca()
        ax.set_xlabel('Threshold', labelpad=24)
        plt.grid(True)
        if legend: plt.legend(loc='upper right')



def makeThrScan(y, predict_test, labels, name, linestyle='-',legend = False):

        fpr, tpr, auc1, thr  = rocData(y, predict_test, labels)

        plotThrScan(fpr,tpr,thr,labels,name,legend=True)


def makeRoc(y, predict_test, labels, name, linestyle='-',legend = False):

        fpr, tpr, auc1, thr  = rocData(y, predict_test, labels)

        plotRoc(fpr, tpr, auc1, labels, linestyle, legend=True)




def EvaluateMetrics(y_test,y_pred,y_hls,name):

        labels = np.array(['zero', 'one', 'two', 'three+'])
        print(labels)


        # ROC
        fig, ax = plt.subplots(figsize=(12, 12))
        makeRoc(y_test, y_pred, labels, name)
        plt.gca().set_prop_cycle(None) # reset the colors
        makeRoc(y_test, y_hls, labels, name,linestyle='--')
        plt.gca().set_prop_cycle(None) # reset the colors
        #makeRoc(y_hls, y_pred, labels, linestyle=':')

        lines = [Line2D([0], [0], ls='-'),
                Line2D([0], [0], ls='--')]
        leg = Legend(ax, lines, labels=['Model', 'HLS emulation'], loc='upper left', frameon=False)
        ax.add_artist(leg)



        #plt.title(title)
        outPdf = FOLDER + 'result/plot/' + name + '_roc.pdf'
        plt.savefig(outPdf, bbox_inches='tight', dpi=300)
        print('%s ready' %outPdf)
        plt.clf()

        # Threshold scan
        fig, ax = plt.subplots(figsize=(12, 12))
        makeThrScan(y_test, y_pred, labels, name)
        plt.gca().set_prop_cycle(None) # reset the colors
        makeThrScan(y_test, y_hls, labels, name, linestyle='--')


        lines = [Line2D([0], [0], ls='-'),
                Line2D([0], [0], ls='--')]
        leg = Legend(ax, lines, labels=['Model', 'HLS emulation'], loc='upper left', frameon=False)
        ax.add_artist(leg)

        outPdf = FOLDER + 'result/plot/' + name +'_thr.pdf'
        plt.savefig(outPdf, bbox_inches='tight', dpi=300)
        print('%s ready' %outPdf)



def OtherMetrics(y_test,y_pred,y_hls, model_name):
        y_pred_1d = transform1d(y_pred)
        y_test_1d = transform1d(y_test)
        y_hls_1d  = transform1d(y_hls)


        print("Accuracy baseline : {}".format(accuracy_score(y_test_1d, y_pred_1d)))
        print("Accuracy hls4ml   : {}".format(accuracy_score(y_test_1d, y_hls_1d)))

