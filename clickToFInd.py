# Made for REU 2018
# Left click on cursor in pictures to crop around it
# Middle click if can't find cursor
# Right click on the cv windows to shift focus
# Escape or Space to quit

import numpy as np
import cv2 as cv
import sys, os

DATA_DIR = 'data'
EXTENSION = '.jpg'
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


pic_Master = 0
leftStart = 0
midStart = 0
rightStart = 0
# I DON'T LIKE DOING IT THIS WAY BUT
# I was too lazy to figure out how to
# get window/image information from
# the event call
PADDING = 32
skipped = False
# crop and delete on LBUTTON
# skip on RBUTTOn
def cropLeft(event, x, y, flags, param):
	global skipped
	if event == cv.EVENT_LBUTTONDOWN:
		skipped = False
		crop(x,y,leftStart)
	elif event == cv.EVENT_MBUTTONDOWN:
		skipped = True
		cv.destroyAllWindows()

def cropMid(event, x, y, flags, param):
	global skipped
	if event == cv.EVENT_LBUTTONDOWN:
		skipped = False
		crop(x,y,midStart)
	elif event == cv.EVENT_MBUTTONDOWN:
		skipped = True
		cv.destroyAllWindows()

def cropRight(event, x, y, flags, param):
	global skipped
	if event == cv.EVENT_LBUTTONDOWN:
		skipped = False
		crop(x,y,rightStart)
	elif event == cv.EVENT_MBUTTONDOWN:
		skipped = True
		cv.destroyAllWindows()

def crop(x,y,leftBoundary):
	global curses
	global EXTENSION
	FILE_NAME = str(curses) + '_' + 'FOUND' + EXTENSION
	curses = curses + 1
	cv.imwrite(os.path.join(CURSE_DIR,FILE_NAME),
			   pic_Master[y-PADDING:y+PADDING,x-PADDING+leftBoundary:x+PADDING+leftBoundary])
	cv.destroyAllWindows()

#get image
#shape = (height, width, channels)
keyInput = 0
for VIDEO in VIDEO_DIRS:
	for BEHVR in BEHVR_FOLDERS[VIDEO]:
		ALL_FRAMES = os.listdir(os.path.join(DATA_DIR,VIDEO,BEHVR))
		for IMAGE in ALL_FRAMES:
			IMAGE_PATH = os.path.join(DATA_DIR,VIDEO,BEHVR,IMAGE)
			_, EXTENSION = os.path.splitext(IMAGE_PATH)
			pic = cv.imread(IMAGE_PATH)
			pic_Master = np.copy(pic)

			height, width, _ = pic.shape

			midStart = width//3
			rightStart=(2*width)//3

			left = pic[:,0:midStart]
			mid = pic[:,midStart:rightStart]
			right = pic[:,rightStart:width]

			cv.namedWindow('LEFT')
			cv.namedWindow('MID')
			cv.namedWindow('RIGHT')
			
			cv.setMouseCallback('LEFT',cropLeft)
			cv.setMouseCallback('MID',cropMid)
			cv.setMouseCallback('RIGHT',cropRight)

			cv.imshow('LEFT',left)
			cv.imshow('MID',mid)
			cv.imshow('RIGHT',right)

			keyInput = cv.waitKey(0)

			# 27 - esc
			# 32 - space
			if keyInput == 27 or keyInput == 32:
				quit()
			#If not quitted, remove image path
			if skipped:
				FILE_NAME = str(non_curses) + '_' + 'BIG' + EXTENSION
				non_curses = non_curses + 1
				cv.imwrite(os.path.join(NO_CURSE_DIR,'BIG',FILE_NAME),pic_Master)
			os.remove(IMAGE_PATH)