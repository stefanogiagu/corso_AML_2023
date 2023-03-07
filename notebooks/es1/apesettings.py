import os
# in case of cocalc project
#FOLDER     ='/projects/ccd019a8-9da4-4bc8-84f6-2ad2c19bb223/hls4ml_cocalc/Rinngs/'
#DATAFOLDER ='/projects/ccd019a8-9da4-4bc8-84f6-2ad2c19bb223/data/NA62/'

# in case of ap3iron4
HOME = os.getenv('HOME')
#FOLDER     = HOME + '/hls4ml_cocalc/Rinngs/'	# Alessandro, 2022 April 15th
FOLDER = HOME + '/rings/hls4ml_cocalc/Rinngs/rinngs/'  # Matteo,  2022 April 19th

print('---------------')
print(FOLDER)
print('---------------')
DATAFOLDER ="/home/andrea/rings/data/"

FOLDER_MODEL  = 'Model'
FOLDER_TRAIN  = 'Train'
FOLDER_QTRAIN = 'QTrain'


def CreateDir(path):
	# Create a directory if it does not exist
	success = os.path.exists(path)
	if not success:
		os.makedirs(path)
