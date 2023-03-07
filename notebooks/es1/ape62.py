import numpy as np

from apesettings import FOLDER

# module-wide constants
MAXHIT  = 64
dx = +170;
dy = +10;



###############
#  Track
# Event data reconstructed offline by DownstreamTrack. It contains information from many detectors included STRAW,RICH,LKr and MUV3
################


class Track:
	def __init__(self,id):
		self.id=id
		# ring fit results
		self.hit=[]   # used hits
		self.x=666.   # coordinate x of the center
		self.y=666.   # coordinate y of the center
		self.r=666.   # radius

		# other detectors
		self.e=-666
		self.p=-666
		self.muv=-666
		self.eop=-666
		self.hypo=-666

	def __del__(self):
		self.hit.clear()

	def dump(self):
		print("TRACK %d "      % self.id , end = ' ')
		print("X %12.3lf "     % self.x  , end = ' ')
		print("Y %12.3lf "     % self.y  , end = ' ')
		print("R %12.3lf "     % self.r  , end = ' ')
		self.hit.sort()
		print("hit list (%2d) " %len(self.hit),end = '')
		#print(self.hit)
		#print("                                                     " , end = '')
		print("HYPO %2d "   % self.hypo, end = '')
		print("LKR %9.3lf "   % self.e  , end = '')
		print("STRAW %9.3lf " % self.p  , end = '')
		print("MUV %9.3lf "   % self.muv, end = '')
		print("EOP %9.3lf "   % self.eop, end = '')
		print()

	def add_hit(self, ch):   #channel
		self.hit.append(ch)

###############
#  ring
# Event data reconstructed offline by RichReco. It contains ring fit results and hit list only
################
class Ring:
    def __init__(self,id): # constructor requires an ID
        self.id=id

        # ring fit results
        self.hit=[]   # used hits
        self.x=666.   # coordinate x of the center
        self.y=666.   # coordinate y of the center
        self.r=666.   # radius

    def __del__(self):
        self.hit.clear()

    def dump(self):
        print("RING  %d "      % self.id , end = '')
        print("X %12.3lf "     % self.x  , end = '')
        print("Y %12.3lf "     % self.y  , end = '')
        print("R %12.3lf "     % self.r  , end = '')
        print("hit list (%2d) " % len(self.hit), end = '')
        self.hit.sort()
        #print(self.hit)
        print()


    def add_hit(self, ch):   #channel
        self.hit.append(ch)

##############3
# Ring GPU
# Event data reconstructed offline by GPU-RICH
################3

class RingGPU:
    def __init__(self,id): # constructor requires an ID
        self.id=id

        # ring fit results
        self.hit=[]   # used hits
        self.x=666.   # coordinate x of the center
        self.y=666.   # coordinate y of the center
        self.r=666.   # radius
        self.method=666

    def __del__(self):
        self.hit.clear()

    def dump(self):
        print("RING GPU %d "      % self.id , end = '')
        print("X %12.3lf "     % self.x  , end = '')
        print("Y %12.3lf "     % self.y  , end = '')
        print("R %12.3lf "     % self.r  , end = '')
        print("Method %d "     % self.method  , end = '')
        print("hit list (%2d) " % len(self.hit),end = '')
        self.hit.sort()
        #print(self.hit)
        print()


    def add_hit(self, ch):   #channel
        self.hit.append(ch)
##################################3
# Event
########################333
# Event: online RICH data + reconstructed offline data
class Event:
    def __init__(self, id):
        self.id=id
        self.hit=[]   # online RICH data
        self.track=[]
        self.ring=[]
        self.ringGPU=[]
        self.eleA=-1;
        self.eleB=-1;
        self.eleC=-1;

    def __del__(self):
        self.hit.clear()
        del self.track[:]
        del self.ring[:]

    def add_hit(self, ch):   #channel
        self.hit.append(ch)

    def add_track(self, a):   #a = an instance of the Track class
        self.track.append(a)

    def add_ring(self, a):   #a = an instance of the Ring class
        self.ring.append(a)

    def add_ringGPU(self, a):   #a = an instance of the Ring class
        self.ringGPU.append(a)

    def set_Ele(self, a,b,c):  # we have 3 definitions of what an electron is !!!
        self.eleA = a
        self.eleB = b
        self.eleC = c

    def dump(self):
        print("**********************")
        print("EVT %d " %self.id)
        print("**********************")
        print("TRACK %2d " % len(self.track))
        print("RING  %2d " % len(self.ring))
        print("GPU   %2d " % len(self.ringGPU))
        print("NELE  %2d " % self.eleA )
        print("NELE  %2d " % self.eleB )
        print("NELE  %2d " % self.eleC )

        print("HIT (%2d) " % len(self.hit), end = ' ')
        for i in range(len(self.hit)):
            print(self.hit[i], end = ' ')
        print()
        for i in range(len(self.track)):
            self.track[i].dump()
        for i in range(len(self.ring)):
            self.ring[i].dump()
        for i in range(len(self.ringGPU)):
            self.ringGPU[i].dump()


###################################################3
# READ Data
####################################################
# file     - INPUT -  filename
# data     - OUTPUT - dictionary to store data for ML
# maxevent - INPUT -  max number of events to be saved in the dictionary from this file
# event    - OUTPUT - a list, each element is an event, full event description, use it for plotting
# maxforplot- INPUT - max number of events to be saved in the list from this file

def read_data_formatRECO(file, data, maxevent, event, maxforplot, offset):
	ev=0 # event counter
	evforPlot=0
	k=0  # in list event counter
	trList = [] # track index i  downstream track reconstruction
	riList = [] # ring  index j  RIch Reco reconbstruction
	gpList = [] # ring  index z  gpu reconstruction
	i=0  # tracks per event
	j=0  # rings per event
	z=0  # rings per event as reconstructed by gpu
	hits_array = np.zeros(MAXHIT)
	gpu_ele_count = -1; # to be checked when GPU reconstructed data will be sistematically available
	ngpuring      = -1  # to be checked when GPU reconstructed data will be sistematically available
	with open(file) as f:
		for line in f:
			#print(line)
			linebuf=line.split()
			time = int(linebuf[0],16) # to fix if needed,  I don't know how to read hex in python
			#burstID = int(linebuf[1])
			eventID = int(linebuf[2])  # caveat, gpu reco data use arbitrary event id while na62 data follow the event ID of the experiment
			tag  = int(linebuf[3])
			# GPU Reco
			if tag ==78:
				gpList.append(RingGPU(z))
				gpList[z].x   = float(linebuf[5])
				gpList[z].y   = float(linebuf[6])
				gpList[z].r   = float(linebuf[7])
				nhit          = int(linebuf[9])
				for h in range(nhit):
					channel = -1;  # hit list is missing from gpu reco output
					gpList[z].add_hit(channel)
				radius = gpList[z].r
				if (radius >185 and radius < 195):
					gpu_ele_count += 1
				#gpList[z].dump()
				z += 1

			# NA62 RichReco
			if tag ==21:
				riList.append(Ring(j))
				riList[j].x   = float(linebuf[5])
				riList[j].y   = float(linebuf[6])
				riList[j].r   = float(linebuf[7])
				nhit          = int(linebuf[9])
				hitlist       = linebuf[10:10+nhit]
				for h in range(len(hitlist)):
					channel = int(hitlist[h])
					riList[j].add_hit(channel)
				#riList[j].dump()
				j += 1

			# NA62 downstream track
			if tag ==20:  # RICH
				trList.append(Track(i))
				trList[i].hypo= int  (linebuf[4])
				trList[i].x   = float(linebuf[5])
				trList[i].y   = float(linebuf[6])
				trList[i].r   = float(linebuf[7])
				nhit          = int  (linebuf[9])
				hitlist       = linebuf[10:10+nhit]
				for h in range(len(hitlist)):
					channel = int(hitlist[h])
					trList[i].add_hit(channel)

			if tag==30:  # LKr
				trList[i].e = float(linebuf[4])

			if tag==40:  # STRAW
				trList[i].p = float(linebuf[7])
				#trList[i].teta = float(linebuf[8])
				#trList[i].phi = float(linebuf[9])

			if tag==50:  # MUV3
				trList[i].muv = float(linebuf[4])

			if tag==60:  # Mixed info
				trList[i].eop = float(linebuf[4])
				#trList[i].dump()
				i += 1

			# SUMMARY
			if tag==22: # Downstream track Summary
				ntracks  = int(linebuf[4])
				el1      = int(linebuf[5]) # <--- number of electrons for downstream track #
				mu       = int(linebuf[6])
				pi       = int(linebuf[7])
				ka       = int(linebuf[8])

			if tag==77: # GPU reconstruction summary
				ngpurings = int(linebuf[4])
				method   = int(linebuf[5])
				ngpuhit  = int(linebuf[9])

			# TAG 23 is the last line of the event!
			if tag==23: # Rich Reco Summary + Custom definition of what an electron is + TDC data (online!)
				nrings   = int(linebuf[4])
				el2      = int(linebuf[5]) # <--- number of electrons for rich reco (geometrical)
				el3      = int(linebuf[6]) # <--- number of electrons for custom definition
				spare    = int(linebuf[7]) # spare
				spare    = int(linebuf[8]) # spare
				nhits    = int(linebuf[9]) # number of hits
				# max hit number is fixed to 64
				if int(nhits)>MAXHIT:
					nhits=MAXHIT
				# read hit list
				listch  = linebuf[10:10+nhits]

				# copy listch in a fixed size array
				for i in range(nhits):
					hits_array[i]  = listch[i]  # channel ID in range 0..2047 + zero filling
					hits_array[i] += 1          # channel ID in range 1..2048 + zero filling, save channel zero from oblivion
				if(0):	# print debug
					print(nhits)
					print(listch)
					print(hits_array)


				# Append data to output lists
				data["np_track"].append(ntracks)
				data["np_reco" ].append(nrings)
				data["np_gpu"  ].append(ngpuring)
				data["ne_eop"  ].append(el3)
				data["ne_track"].append(el2)
				data["ne_reco" ].append(el1)
				data["ne_gpu"  ].append(gpu_ele_count)
				data['hitlist' ].append(listch)

				# Add the event to the list and populate the event with hitlist, tracks and rings
				# TODO: check that if reading multiple files the events in the list are not overwritten
				if ev < maxforplot:
					event.append(Event(ev+offset))
					event[ev+offset].set_Ele(el1,el2,el3)
					for m in range (len(listch)):
						event[ev+offset].add_hit(listch[m])
					for m in range (len(trList)):
						event[ev+offset].add_track(trList[m])
					for m in range (len(riList)):
						event[ev+offset].add_ring(riList[m])
					for m in range (len(gpList)):
						event[ev+offset].add_ringGPU(gpList[m])
					evforPlot +=1
                    			#event[ev].dump()


				#if(ev%1000==0):
				if(ev%10000==0):
				#if(ev%100000==0):
					print("EV %8d" %ev, end = ' ')
					print("(TRACK,RICH,GPU)", end = ' ')
					print("N_PARTICLES", end = ' ')
					print("%d " %ntracks, end = ' ')
					print("%d " %nrings, end = ' ')
					print("%d " %ngpuring, end = ' ')
					print("N_ELECTRONS", end = ' ')
					print("%d " %el1, end = ' ')
					print("%d " %el2, end = ' ')
					print("%d " %el3, end = ' ')
					print("%d " %gpu_ele_count, end = ' ')
					print()

				# increase event counter
				ev += 1

				# reset event variables
				i=0
				j=0
				z=0
				trList = []
				riList = []
				gpList = []
				gpu_ele_count = 0

				# check enough statistics was collected
				if ev < maxevent:
					# reset event variables (get ready for new event data)
					hits_array=np.zeros(MAXHIT)

				else:
					print("Finished with this file");
					#print("file[",idx,"] evt = ",evt," totrings = ",totrings)
#					print("Events %d" % ev)
					return ev, evforPlot
#                    			break
#	print("Events %d" % ev)
	return ev,evforPlot



#######################################################################################
# RICH geometry  (electronic channel <--> x,y coordinates of the associated PMT)
#######################################################################################

def getChannelCoordinates():
	# arbitrary translation to have symmetric x and y axis on the datadisplay
#	dx= +170;
#	dy= +10;
	filemap = FOLDER + '/rinngs/RICH_map_corr_2017.data'

	x_pmt=[]
	y_pmt=[]
	echan=[]

	counter=0

	with open(filemap) as f:
    		for line in f:
        		line=line.strip().split()
        		if(float(line[1])<5000):
            			eleid = int   ( line[0] )
            			x     = float ( line[1] )
            			y     = float ( line[2] )
            			#print("ID %4d"%eleid,"X %6.3lf" %x, "Y %6.3lf" %y,)
            			x_pmt.append( x )
            			y_pmt.append( y )
            			echan.append( eleid )

	return x_pmt, y_pmt, echan

def getChannelCoordinatesNormalised():

	N=64  # image size is N x N    # this must become an argument

	x_pmt,y_pmt,echan = getChannelCoordinates()

	# transform lists to numpy arrays
	x_pmt=np.array(x_pmt)
	y_pmt=np.array(y_pmt)

	# PMT Coordinates normalized  in range 0 to 1
	rangeX = np.max(x_pmt) -  np.min(x_pmt)
	rangeY = np.max(y_pmt) -  np.min(y_pmt)

	x = (x_pmt - np.min(x_pmt)) / rangeX
	y = (y_pmt - np.min(y_pmt)) / rangeY

	x=(np.round((N-1)*x)).astype('int16')
	y=(np.round((N-1)*y)).astype('int16')        

	return x,y
 
def to_image(x,y,hitlist=np.arange(1952,dtype='int16')):

	N=64  # image size is N x N # this must become an argument
	I=np.zeros([N,N])
	for i in hitlist:
		I[x[i],y[i]]+=1
     
	return I




# to be moved elsewhere
#print('len x :{}, max x :{}, min x :{}'.format(len(x_pmt),np.max(x_pmt),np.min(x_pmt)))
#x_pmt=np.array(x_pmt)
#y_pmt=np.array(y_pmt)





######################################################
# RICH EVENT DISPLAY
##########################################

import matplotlib.pyplot as plt
def custom_plot(opt,evt, ax=None, **plt_kwargs):

    # size of the pixels in the display
    r_pmt = 9.0
    r_pmt_reduced=9.0

    # get coordinates
    x_pmt, y_pmt, echan = getChannelCoordinates()

    if(0):  # pixel  arrray  (time consuming, 10 times slower! 6 seconds vs 0.6 seconds)
        print("opt %d " % opt, "pmt %d" % len(x_pmt))
        for i in range(len(x_pmt)):
            x = float(x_pmt[i]) + dx
            y = float(y_pmt[i]) + dy
            if (float(x)<2000):
                pixel = plt.Circle( (x,y), r_pmt, facecolor='none', edgecolor='black', lw=0.1)
                ax.add_artist(pixel)

    if(1): # hit pattern (as online except that saleve and jura use different colors)
        size= len(evt.hit)
        for i in range(size):
            hit= int(evt.hit[i])
            # id=-1
            x = -300
            y = +300   # fake pixel to avoid errors due a map not up to date
            for e in range (1952):
                if(hit==echan[e]):
                    x = float(x_pmt[e]) + dx
                    y = float(y_pmt[e]) + dy
            color = 'm' # saleve
            if(hit<1024):
                color = 'y' # jura
            circle= plt.Circle( (x,y), r_pmt, facecolor=color, edgecolor='black', lw=0.1)
            ax.add_patch(circle)
            #plt.text(x+9,y+9, hit, fontsize=9) # channel ID print near the corresponding PMT

    if(1): # fit results
        # read the list of fit results, it can be track list or ring list
        if(opt==0):
            myList = evt.track
        elif(opt==1):
            myList = evt.ring
        elif(opt==2):
            myList = evt.gpuring

        # loop on tracks/rings
        for i in range(len(myList)):
            # read fit results
            xc = myList[i].x + dx
            yc = myList[i].y + dy
            rc = myList[i].r
            #myList[i].dump() # print event on screen
            # color code based on PID hypo for downstreamtrack and on radius for RichReco
            if(opt==0):
                hypo = myList[i].hypo
                color = 'g'   # multiple hypo is magenta
                if(hypo==0):   # background hypo is yellow
                    color = 'y'
                elif(hypo==1): # electron hypo is blue
                    color = 'b'
                elif(hypo==2):  # muon hypo is red
                    color = 'r'
                elif(hypo==3):  # pion hypo is black
                    color = 'k'
                elif(hypo==4): # kaon hypo
                    color = 'g'
            if(opt==1):
                color = 'k' # default
                if(rc >= 185):   # electron radius is blue
                    color = 'b'
                if(rc >= 195): # bad fit is green
                    color = 'g'
            if(opt==2):
                color = 'k' # default
                if(rc >= 185):   # electron radius is blue
                    color = 'b'
                if(rc >= 195): # bad fit is green
                    color = 'g'

            # retrieve hit list used for this particular fit
            nhit = len(myList[i].hit)
            for j in range(nhit):
                hit= int(myList[i].hit[j])
                x = -300
                y = +300   # fake pixel to avoid errors due a map not up to date
                for e in range (1952):
                    if(hit==echan[e]):
                        x = float(x_pmt[e]) + dx
                        y = float(y_pmt[e]) + dy
                pixel= plt.Circle( (x,y), r_pmt_reduced, facecolor=color, edgecolor='k', lw=0.1)
                ax.add_patch(pixel)
            # plot fit results
            if(xc<1000):
                ring=plt.Circle((xc,yc), rc, facecolor='none', edgecolor=color, lw=2.8)
                ax.add_patch(ring)
    return(ax)





