import cv2 as cv
import numpy as np
from keras.models import Model, model_from_json
from keras.applications.vgg16 import VGG16
from keras.layers import Input
#from keras.preprocessing import image
import keras
import json
import time
import sys, os

def natClamp(inp,max):
	if inp < 0:
		return 0
	if inp > max:
		return max
	return inp

architecture = 'FC-32-rms'
PREV_WEIGHT_VERSION = 38

model = model_from_json(json.load(open('ARCH_'+architecture+'.json')))
model.load_weights('WEIGHTS_'+architecture+'_sgdFT-'+str(PREV_WEIGHT_VERSION)+'.h5')

DATA_DIR = 'data'
CURSE_DIR = os.path.join(DATA_DIR,'TIMELI_curse')
curses = len(os.listdir(CURSE_DIR))
print('curses: ' + str(curses))
NO_CURSE_DIR = os.path.join(DATA_DIR,'TIMELI_nocurse')
non_curses = len(os.listdir(NO_CURSE_DIR))
print('non_curses: ' + str(non_curses))

VIDEO_DIRS = os.listdir(DATA_DIR)
print(VIDEO_DIRS)
VIDEO_DIRS = [d for d in VIDEO_DIRS if str.startswith(d,'video')]
BEHVR_FOLDERS = {}
for VIDEO in VIDEO_DIRS:
	BEHVR_FOLDERS[VIDEO] = os.listdir(os.path.join(DATA_DIR,VIDEO))
	print(BEHVR_FOLDERS[VIDEO])



#get image
for VIDEO in VIDEO_DIRS:
	for BEHVR in BEHVR_FOLDERS[VIDEO]:
		ALL_FRAMES = os.listdir(os.path.join(DATA_DIR,VIDEO,BEHVR))
		print(ALL_FRAMES)
		for IMAGE in ALL_FRAMES:
			IMAGE_PATH = os.path.join(DATA_DIR,VIDEO,BEHVR,IMAGE)
			_, EXTENSION = os.path.splitext(IMAGE_PATH)
			pic = cv.imread(IMAGE_PATH)
			pic_Master = np.copy(pic)

			_, inpHeight, inpWidth, _ = model.inputs[0].shape
			strideH = inpHeight // 2
			strideW = inpWidth // 2

			picHeight, picWidth, _ = pic.shape
			monitorWidth = picWidth // 3
			totalStridesH = picHeight//strideH
			totalStridesW = picWidth//strideW
			likelihoods = np.zeros((totalStridesH,totalStridesW),dtype=np.float)

			pre_prediction_time = time.time()

			#Find mouse likelihoods
			for h in range(0,totalStridesH):
				for w in range(0,totalStridesW):
					topLeftH = h * strideH
					topLeftW = w * strideW
					if topLeftH + inpHeight > picHeight:
						topLeftH = picHeight - inpHeight
					if topLeftW + inpWidth > picWidth:
						topLeftW = picWidth - inpWidth
					sample = np.zeros((1,48,48,3),dtype=pic.dtype)

					sample[0] = pic[topLeftH:topLeftH+inpHeight, topLeftW:topLeftW+inpWidth]
					prediction = model.predict(sample)
					likelihoods[h,w]=prediction[0][0]
				print('finished pass h='+str(h))
				

			post_prediction_time = time.time()
			prediction_time = post_prediction_time - pre_prediction_time
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print('Total prediction time: ' + str(prediction_time))
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

			#j - 106 - negative
			#k - 107 - show Master
			#space - 32 - quit
			#o - 111 - next Desktop pic
			#i - 105 - positive

			#declare for persistence
			k = 0
			maxLocUL = 0
			maxLocDR = 0
			PADDING = 4
			inputKey = 0
			while inputKey != 105 and inputKey != 111:

				coords = np.unravel_index(np.argmax(likelihoods, axis=None), likelihoods.shape)
				k = k-1
				likelihoods[coords]=k
				maxLocUL = (int(coords[0] *strideH),int(coords[1]*strideW))
				maxLocDR = (int(maxLocUL[0] + inpHeight), int(maxLocUL[1]+inpWidth))
				pic_copy = np.copy(pic_Master)	#opportunity for optimization: only copy necessary area then remath rect
				cv.rectangle(pic_copy,(maxLocUL[1],maxLocUL[0]),(maxLocDR[1],maxLocDR[0]),(0,0,255),3)
				#print(str(natClamp(maxLocUL[0]-15,picHeight))+':'+str(natClamp(maxLocDR[0]+15,picHeight)))
				#print(str(natClamp(maxLocUL[1]-15,picWidth))+':'+str(natClamp(maxLocDR[1]+15,picWidth)))
				cv.imshow('Cropped Pic',pic_copy[natClamp(maxLocUL[0]-15,picHeight):natClamp(maxLocDR[0]+15,picHeight),	natClamp(maxLocUL[1]-15,picWidth):natClamp(maxLocDR[1]+15,picWidth)])
				inputKey = cv.waitKey(0)

				if inputKey == 107:
					for i in range(3):
						cv.imshow('Master_'+str(i),pic[:,i*monitorWidth:(i+1)*monitorWidth])
					inputKey = cv.waitKey(0)					
				cv.destroyAllWindows()
				if inputKey == 106:
					#save negative
					FILE_NAME = str(non_curses) + '_' + str(-1*k) + EXTENSION
					non_curses = non_curses + 1
					cv.imwrite(os.path.join(NO_CURSE_DIR,FILE_NAME),pic_Master[maxLocUL[0]-PADDING:maxLocDR[0]+PADDING,maxLocUL[1]-PADDING:maxLocDR[1]+PADDING])
				elif inputKey == 32:
					if k < -5:
						os.remove(IMAGE_PATH)
					quit()
				if k%10==0:
					print('k = ' + str(k))
						

			if inputKey == 105:
				#save positive
				FILE_NAME = str(curses) + '_' + str(-1*k) + EXTENSION
				curses = curses + 1
				cv.imwrite(os.path.join(CURSE_DIR,FILE_NAME),pic_Master[maxLocUL[0]-PADDING:maxLocDR[0]+PADDING,maxLocUL[1]-PADDING:maxLocDR[1]+PADDING])

			os.remove(IMAGE_PATH)
			#os.unlink(IMAGE_PATH) #same, but using Linux terminology
			print(k)