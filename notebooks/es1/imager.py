import numpy as np

from apesettings import FOLDER
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
