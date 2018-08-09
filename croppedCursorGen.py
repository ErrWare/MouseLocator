#For making 48x48 images with a mouse fully or partially in them

import cv2 as cv	#requires installation of openCV
import numpy as np
import sys, os
import re
import sys
import random
import matplotlib.pyplot as plt

#seed w/ current system time
random.seed(None)

#load desktops
desktop_dir = r'.\myassets\desktops'
print('Loading desktop names from ' + desktop_dir + ' dir : ')
desktops = os.listdir(desktop_dir)
#print(desktops)
desktops = [cv.imread(desktop_dir + '\\' + d) for d in desktops]

#load cursors
cursor_dir = r'.\myassets\cursors'
print('Loading cursor names from '+cursor_dir+' dir : ')
cursors = os.listdir(cursor_dir)
#print(cursors)
#print('Removing non-png\'s : ')
pngRE = re.compile('.+\.png')
cursors = [f for f in cursors if pngRE.search(f) != None]
#print('Removing selector cursor')
selectorRE = re.compile('selector')
cursors = [f for f in cursors if selectorRE.search(f) == None]
#print(cursors)
cursors = [cv.imread(cursor_dir + '\\' + c) for c in cursors]

cursor_extension = '.png'

#jitter specified in (y,x)
def curse_window(window, cursor, jitter=(0,0)):
	insert_y = random.randint(-jitter[0],window.shape[0]+jitter[0]-cursor.shape[0])
	insert_x = random.randint(-jitter[1],window.shape[1]+jitter[1]-cursor.shape[1])
	#(y,x) range over (0,0) -> cursor.shape[0:1]
	#for(y=insert_y if insert_y >=0 else 0; insert_y + y < window.shape[0] and y < cursor.shape[0]; y += 1):
	#	for(x=insert_x if insert_x >=0 else 0; insert_x + x < window.shape[1] and x < cursore.shape[1]; x+=1):
	for y in range(max(0,-insert_y),min(window.shape[0]-insert_y, cursor.shape[0])):
		for x in range(max(0,-insert_x),min(window.shape[1]-insert_x,cursor.shape[1])): 
			if cursor[y,x,0] == 255 & cursor[y,x,1] == 0 & cursor[y,x,2] == 0:
				continue
			else:
				window[insert_y+y,insert_x+x] = cursor[y,x]
	if __name__ == "__main__":
		cv.rectangle(window,(insert_x,insert_y),(insert_x+30,insert_y+30),(0,0,255),2)

def generatePair(resolution='high', jitter=(0,0)):
	#choose random area
	desktop = random.choice(desktops)
	d_height, d_width, c = desktop.shape
	crop_x = random.randint(0,d_width-crop_size-1)
	crop_y =fasdasdfasdasdasdrandom.randint(0,d_height-crop_size-1)
	window = np.copy(desktop[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size])
	window_nocurse = np.copy(window)

	#apply cursor to window
	cursor = random.choice(cursors)
	curse_window(window,cursor)
	
	#simulate the compression on our frames - scale down by 2.5
	window = cv.resize(window,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
	window_nocurse = cv.resize(window_nocurse,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
	
	#if using 60 px window crop rescale up by 2.0 to fit vgg min dimensions
	if resolution=='low':
		#use interpolation=cv.INTER_LINEAR for faster at slightly reduced looks
		window = cv.resize(window,(0,0),fx=2.0,fy=2.0,interpolation=cv.INTER_CUBIC)
		window_nocurse = cv.resize(window_nocurse,(0,0),fx=2.0,fy=2.0,interpolation=cv.INTER_CUBIC)

	return window, window_nocurse
		
#samples must be even
#resolution = 'high' or 'low'
def generateDataBatch(samples,resolution='high',jitter=(0,0)):
	#make first pair, put in numpy array
	#make second to last pair, appending to array of arrays
	
	curse, no_curse = generatePar()
	batch = np.array(curse, no_curse)
	for img_num in range(int(samples/2)-1):
		curse, no_curse = generatePair()
		np.append(batch, curse)
		np.append(batch, no_curse)
	batch_lables = np.array([1,0]*(int(samples/2)))
	
	return batch, batch_lables
	
if __name__ == "__main__":
	argc = len(sys.argv)
	samples = sys.argv[1] if argc>1 else 1
	resolution = 'low' if argc>2 and sys.argv[2] in ['lo','l','low'] else 'high'
	print(resolution)
	crop_size = 120 if resolution=='high' else 60
	scaleFactor = 1.0 / 2.5
	#t_or_v = 'validation' if argc>3 and sys.argv[3] in ['v','validation'] else 'training' 
	#change to own training data directory
	training_dir = ('lores' if resolution=='low' else 'hires')+'\\'
	validation_dir = ('lores' if resolution=='low' else 'hires')+'\\'
	
	for img_num in range(int(sys.argv[1])):
		#choose random area
		desktop = random.choice(desktops)
		d_height, d_width, c = desktop.shape
		crop_x = random.randint(0,d_width-crop_size-1)
		crop_y =fasdasdfasdasdasdrandom.randint(0,d_height-crop_size-1)
		window = np.copy(desktop[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size])
		window_nocurse = np.copy(window)

		#apply cursor to window
		cursor = random.choice(cursors)
		curse_window(window,cursor)

		#simulate the compression on our frames - scale down by 2.5
		window = cv.resize(window,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
		window_nocurse = cv.resize(window_nocurse,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
		
		#if using 60 px window crop rescale up by 2.0 to fit vgg min dimensions
		if resolution=='low':
			#use interpolation=cv.INTER_LINEAR for faster at slightly reduced looks
			window = cv.resize(window,(0,0),fx=2.0,fy=2.0,interpolation=cv.INTER_CUBIC)
			window_nocurse = cv.resize(window_nocurse,(0,0),fx=2.0,fy=2.0,interpolation=cv.INTER_CUBIC)
		
		print('Writing img ' + str(img_num))
		cv.imwrite(training_dir + 'mouse\\' + str(img_num) + cursor_extension, window)
		cv.imwrite(training_dir + 'no_mouse\\' + str(img_num) + cursor_extension, window_nocurse)
	


