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


from apeml import getData	# Data Preprocess
from apeml import EventPlot	# Data Preprocess, closer to getData during development, to be moved in ape62
from apeml import EventPrint  	# Data Preprocess, closer to getData during development, to be moved in ape62
from apeml import ShuffleSplit	# Data Preprocess
from apeml import SaveAll	# Data Preprocess

from imager import Mapping


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


import numpy as np

from apesettings import FOLDER
from apesettings import CreateDir

FOLDER = "data/"
def dataPreprocess(dataset = '2450206', model = "Dense", size = 64, label = "np_track", shuffle = 1):

	lsize = size

	print('Getting data for label %s' % label)
	x,y = getData(dataset,label)

	EventPlot(dataset)
	EventPrint(dataset)

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

