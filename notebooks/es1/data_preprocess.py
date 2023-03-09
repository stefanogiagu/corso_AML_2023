#/*
#* <This code is a part of the NA62 collaboration>
#* Copyright (C) <2019-2023>  <Ape group INFN roma + Andrea>
#*
#* This program is free software: you can redistribute it and/or modify
#* it under the terms of the GNU General Public License as published by
#* the Free Software Foundation, either version 3 of the License, or
#* (at your option) any later version.
#*
#* This program is distributed in the hope that it will be useful,
#* but WITHOUT ANY WARRANTY; without even the implied warranty of
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#* GNU General Public License for more details.
#*
#* You should have received a copy of the GNU General Public License
#* along with this program.  If not, see <http://www.gnu.org/licenses/>.
#*/

from sklearn.model_selection import train_test_split # used in data preprocessing
from tensorflow.keras.utils import to_categorical # used by get data to transform unidimensional target in a 4 dim  vector (one dimension per class)
import json
import os
#from apeml import getData	# Data Preprocess
#from apeml import EventPlot	# Data Preprocess, closer to getData during development, to be moved in ape62
#from apeml import EventPrint  	# Data Preprocess, closer to getData during development, to be moved in ape62
#from apeml import ShuffleSplit	# Data Preprocess
#from apeml import SaveAll	# Data Preprocess

#from imager import Mapping


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


import numpy as np


def CreateDir(path):
	# Create a directory if it does not exist
	success = os.path.exists(path)
	if not success:
		os.makedirs(path)

FOLDER = "data/"
N_LABEL=4  # number of output classes
MAXLABEL=4

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






def dataPreprocess(dataset = '2450206', model = "Dense", size = 64, label = "np_track", shuffle = 1):

	lsize = size

	print('Getting data for label %s' % label)
	x,y = getData(dataset,label)

	#EventPlot(dataset)
	#EventPrint(dataset)

	print("X_SHAPE")
	print(x.shape)
	print("X_SHAPE[0]")
	print(x.shape[0])
	print("Y_SHAPE")
	print(y.shape)

	print("Preprocessing data from dataset "+ dataset +" for model " + model + "of size ", size)


	#####
	#TEMPORARY CODE FOR DEBUG
	#TO BE DECIDED IF THE PREPROCESSING HAS TO BE DONE IN GETDATA or here
	#################

	####


	if model == "Dense":
		print("Data preprocessinG DENSE")
		print("Before normalization")
		print(x[0])	
		x = x /2048.
		print("After normalization")
		print(x[0])	

		
	if model == "Conv":
		# Create an image list from hitlist x
		mapFile = 'data' + '/RICH_map_corr_2017.data'
		imger = Mapping(mapFile,lsize)
		imagelist = []
		nevents = x.shape[0]
		for i in range(nevents):
			image = imger.convert(x[i])
			imagelist.append(image)
			if i%50000==0:
				print("processing image %6d" %i)
		print("Images shape:")
		print(imagelist[0].shape)
		print("Number of images:")
		print(len(imagelist))
		if(1): # optionally create a pdf with few images just created
			# PDF output
			outPdf = 'results/preprocess/' +'prova' + str(size) + '.pdf'
			plt.figure()
			with PdfPages(outPdf) as pdf:
				for i in range(10):
					plt.imshow(imagelist[i])
					pdf.savefig(bbox_inches='tight', dpi=300)
					plt.close()
			print('File %s ready' % outPdf)

		# print statistics and errors
		imger.stat()


		# transform the image list back to numpy array for seamless continuation of the workflow
		x = np.array(imagelist)


	SaveAll(x,y,dataset)
	return

MAXPMT=1952
MAXELE=2048



class Mapping:

	def __init__(self, mapFile,N=64):
		self.N = N					# image size is N x N
		self.filemap = mapFile				# translation table ELEID to spatial coordinates X and Y
		# MAXPMT = 1952
		self.ch = np.zeros(shape=MAXPMT, dtype=int)
		self.x  = np.zeros(shape=MAXPMT, dtype=float)
		self.y  = np.zeros(shape=MAXPMT, dtype=float)
		self.xn = np.zeros(shape=MAXPMT, dtype=float)
		self.yn = np.zeros(shape=MAXPMT, dtype=float)
		# MAXELE = 2048 this allows the hit coordinate addressing i.e. The coordinates of pmt 666 readout by  channel 2031 are at address 2031  

		self.pmt = np.zeros(shape=MAXELE, dtype=int)
		self.xb = np.zeros(shape=MAXELE, dtype=int)
		self.yb = np.zeros(shape=MAXELE, dtype=int)

		# assign -1 as a reset value
		self.xb  = self.xb -1
		self.yb  = self.yb -1
		self.pmt = self.pmt -1


		self.read()
		self.process()
		self.dump()
		self.export()

		self.histoChannel = np.zeros(shape=MAXELE, dtype=int)
		self.histoError   = np.zeros(shape=MAXELE, dtype=int)

	def export(self):

		name = 'imagerMap_human.txt'
		name = 'results/display/' + name


		with open(name, 'w') as f:
			for i in range(MAXELE):
				print("ELE %4d"  % i          , end = ' ', file= f)
				print("PMT %4d"  % self.pmt[i], end = ' ',file= f)
				print("Xb %3d"   % self.xb[i] , end = ' ',file= f)
				print("Yb %3d"   % self.yb[i] , end = '\n',file= f)

		print('Imager map ready at %s (human format)' % name)



		name = 'imagerMap.txt'
		name = 'results/display/' + name

		with open(name, 'w') as f:
			for i in range(MAXELE):
				print("%4d"  % i          , end = ' ', file= f)
				print("%3d"  % self.xb[i] , end = ' ',file= f)
				print("%3d"  % self.yb[i] , end = '\n',file= f)
		print('Imager map ready at %s (computer format)' % name)




	def stat(self):

		name = 'imagerCounts.txt'
		name = 'results/display/' + name

		with open(name, 'w') as f:
			for i in range(MAXELE):
				print("ELE %4d"    % i                    , end = '  ', file= f)
				print("COUNTS %6d" % self.histoChannel[i] , end = '  ', file= f)
				print("ERRORS %6d" % self.histoError[i]   , end = '\n', file= f)

		print('Imager statistics ready at %s' % name)


	def dump(self):
		print("FILE MAP: %s" % self.filemap)
		print(self.x)
		print(self.y)
		print(self.ch)
		print(self.xn)
		print(self.yn)
		print(self.xb)
		print(self.yb)

		if(0):
			for i in range(MAXPMT):
				print("PMT %4d"  % i          , end = ' ')
				print("ELE %4d"  % self.ch[i] , end = ' ')
				print("X %8.3f"  % self.x[i]  , end = ' ')
				print("Y %8.3f"  % self.y[i]  , end = ' ')
				print("Xn %9.6f" % self.xn[i] , end = ' ')
				print("Yn %9.6f" % self.yn[i] , end = ' ')
				print("Xb %3d"   % self.xb[i] , end = ' ')
				print("Yb %3d"   % self.yb[i] , end = '\n')

			for i in range(MAXELE):
				print("ELE %4d"  % i          , end = ' ')
				print("PMT %4d"  % self.pmt[i], end = ' ')
				print("Xb %3d"   % self.xb[i] , end = ' ')
				print("Yb %3d"   % self.yb[i] , end = '\n')






	def read(self):
		with open(self.filemap) as f:
			i=0
			for line in f:
				line=line.strip().split()
				if(float(line[1])<5000):
					self.ch[i] = int   ( line[0] )
					self.x[i]  = float ( line[1] )
					self.y[i]  = float ( line[2] )
					if(0):
						print("%4d %4d %8.3f %8.3f" %(i, self.ch[i], self.x[i], self.y[i]))
					idx = self.ch[i]
					self.pmt[idx]  = i
					i += 1


	def process(self):

		n = self.N
		print("Image %d x %d  = %d pixels"   % (n,n,n*n))

		# X coordinate 
		min   = np.min(self.x)
		max   = np.max(self.x)
		range = max - min
		bin   = range/n

		print("X "                       , end = ' ' )
		print("max %6.3lf "  % max       , end = ' ' )
		print("min %6.3lf "  % min       , end = ' ' )
		print("range %6.3lf" % range     , end = ' ' )
		print("bin %6.3lf "  % bin       , end = '\n')
		self.xn = (self.x - min) / range

		# Y coordinate
		min = np.min(self.y)
		max = np.max(self.y)
		range = max - min
		bin   = range/n

		print("Y "                       , end = ' ' )
		print("max %6.3lf "  % max       , end = ' ' )
		print("min %6.3lf "  % min       , end = ' ' )
		print("range %6.3lf" % range     , end = ' ' )
		print("bin %6.3lf "  % bin       , end = '\n')
		self.yn = (self.y - min) / range




		# Bin
		maxbin     = self.N-1
		i = 0
		for val in self.xn:
			binFloat   = maxbin*val
			binRounded = np.round(binFloat)
			binInteger = (binRounded).astype('int16')
#			print("X  "                , end = ' ')
#			print("%4d "   % i          , end = ' ')
#			print("%4d "   % self.ch[i] , end = ' ')
#			print("%8.3lf" % self.xn[i] , end = ' ')
#			print("%8.3lf" % binFloat   , end = ' ')
#			print("%8.1lf" % binRounded , end = ' ')
#			print("%3d"    % binInteger , end = '\n')
			ele = self.ch[i]
			self.xb[ele] = binInteger
			i += 1

		i = 0
		for val in self.yn:
			binFloat   = maxbin*val
			binRounded = np.round(binFloat)
			binInteger = (binRounded).astype('int16')
#			print("Y  "                , end = ' ')
#			print("%4d "   % i          , end = ' ')
#			print("%4d "   % self.ch[i] , end = ' ')
#			print("%8.3lf" % self.yn[i] , end = ' ')
#			print("%8.3lf" % binFloat   , end = ' ')
#			print("%8.1lf" % binRounded , end = ' ')
#			print("%3d"    % binInteger , end = '\n')
			ele = self.ch[i]
			self.yb[ele] = binInteger
			i += 1

		i = 0
		for val in self.xb:
#			print("%4d " % i         , end = ' ' )
#			print("%3d " % val       , end = ' ' )
#			print("%3d " % self.yb[i], end = '\n' )
			i += 1



	def convert(self,hitlist=np.arange(MAXELE,dtype='int16')):
		n = self.N
		I = np.zeros([n,n])

		for val in hitlist:
			ele = int   ( val )	# integer casting because the channel ID is used for addressing		
#			print("ELE %d " % ele, end  = ' ');

			if(ele <=0):  # ele>0 kill 1 pixel (pixel 0 and default values coincides, should do +1 but... neglect for the moment)
				continue


			x = self.xb[ele]
			y = self.yb[ele]
			if(x>=0 and y>=0):  # ele>0 kill 1 pixel (pixel 0 and default values coincides, should do +1 but... neglect for the moment)

				I[x,y] += 1
				# update counting statistics
				self.histoChannel[ele] +=1

				if(0):
					print("%8d" % ele , end = ' ' )
					print("%8d" % x   , end= ' ' )
					print("%8d" % y   , end= '\n' )
			else:
				self.histoError[ele] +=1
				#print("Error: unconnected channels found in the hit list %d" % ele)

		return I

###################################
import sys
dataset  = sys.argv[1]
model    = sys.argv[2]
size     = int(sys.argv[3])
label    = sys.argv[4]
shuffle  = int(sys.argv[5])


print('*************************')
print('*  Data Pre-Processing  *')
print('*************************')
print('Dataset = %s' % dataset)
print('Model   = %s' % model)
print('Size    = %d' % size)
print('Label   = %s' % label)
print('Shuffle = %d' % shuffle)


dataPreprocess(dataset, model, size, label, shuffle)

