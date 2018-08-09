#Made for generating data for cursor finding
#Look at the comments for usage notes
#to make these importable modules put them in the Python36 (or w/e version you have) folder
import cv2 as cv	#requires installation of openCV
import numpy as np
import sys, os
import re
import sys
import random
import matplotlib.pyplot as plt

#Load assets
#print('Loading window names from .\\windows dir : ')
windows = os.listdir(r'.\myassets\windows')
#print(windows)

#print('Loading taskbar names from .\\taskbars dir : ')
taskbars = os.listdir(r'.\myassets\taskbars')
#print(taskbars)

#print('Loading cursor names from .\\cursors dir : ')
cursors = os.listdir(r'.\myassets\cursors')
#print(cursors)
#print('Removing non-png\'s : ')
pngRE = re.compile('.+\.png')
cursors = [f for f in cursors if pngRE.search(f) != None]
#print(cursors)

#print('Loading wallpapers from .\\wallpapers dir : ')
wallpapers = os.listdir(r'.\myassets\wallpapers')
#print(wallpapers)


def addCursor(base, copy_masked):
	
	#please optimize me
	copy_height, copy_width, q = copy_masked.shape
	base_height, base_width, b = base.shape
	topLeft_y = random.randint(0,base_height-copy_height-1)
	topLeft_x = random.randint(0,base_width-copy_width-1)
	for y in range(0, copy_height):
		for x in range(0, copy_width):
			if copy_masked[y,x,0] == 255 & copy_masked[y,x,1] == 0 & copy_masked[y,x,2] == 0:
				continue
			else:
				base[topLeft_y+y,topLeft_x+x] = copy_masked[y,x]
				
	#cv.rectangle(base,(topLeft_x,topLeft_y),(topLeft_x+30,topLeft_y+30),(0,0,255),2)
	return (topLeft_x, topLeft_y)
				
def addWindow(base, outline, window):
	outline_width = 4
	base_height, base_width, c = base.shape
	topLeft_y = random.randint(0,base_height)
	topLeft_x = random.randint(0,base_width)
	_, _, outline_channels = outline.shape()
	#make sure no copy stays w/in dims(necessary?)
	window_height, window_width, c = window.shape
	copy_height = min(base_height, topLeft_y + window_height) - topLeft_y
	copy_width = min(base_width, topLeft_x + window_width) - topLeft_x
	if(copy_height > 0 and copy_width > 0):
		base[topLeft_y:topLeft_y+copy_height-1,topLeft_x:topLeft_x+copy_width-1] = window[0:copy_height-1,0:copy_width-1]
		cv.rectangle(outline,(topLeft_x,topLeft_y),(topLeft_x+copy_width-1,topLeft_y+copy_height-1),255 if outline_channels==1 else [255,255,255],-1)
		cv.rectangle(outline,(topLeft_x+outline_width,topLeft_y+outline_width),(topLeft_x+copy_width-1-outline_width,topLeft_y+copy_height-1-outline_width),0 if outline_channels==1 else [0,0,0],-1)

def addTaskBar(base, outline, taskbar, screen):
	t_height, t_width, c = taskbar.shape
	b_height, b_width, c = base.shape
	_, _, outline_channels = outline.shape()
	#print(str((1920*(screen-1))+b_width-t_width) + ' : ' +str((1920*(screen-1))+b_width))
	base[b_height-t_height:b_height,(1920*(screen))-t_width:(1920*screen)] = taskbar[:,:]
	outline[b_height-t_height:b_height,(1920*(screen))-t_width:(1920*screen)] = 0 if outline_channels==1 else [0,0,0]

def makeDesktop(desktopScreens=3, windows_generated=12, outlineChannels=1):
	#Specify screen height and width in pixels
	#Has to match the dimensions of the wallpaper images
	SCREEN_HEIGHT = 1080
	SCREEN_WIDTH = 1920

	desktopScreens = 3
	desktopScreenPrimary = 2
	
	desktops = np.zeros((desktopScreens,SCREEN_HEIGHT,SCREEN_WIDTH,3),dtype=np.uint8)
	desktop = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH*desktopScreens,3),dtype=np.uint8)
	#outline would ideally have 1 channel, but I don't know how to 
	#inflate that in order to merge them for the together picture
	outline = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH*desktopScreens,outlineChannels),dtype=np.uint8)
	for i in range(0,desktopScreens):
		desktops[i] = cv.imread('.\\myassets\\wallpapers\\' + wallpapers[random.randint(0,len(wallpapers)-1)])
		desktop[0:SCREEN_HEIGHT,SCREEN_WIDTH*i:SCREEN_WIDTH*(i+1)] = desktops[i,:,:]
		
	#start making windows
	for i in range(0,windows_generated):
		#print('adding window ' + str(i))
		window = cv.imread('.\\myassets\\windows\\' + windows[random.randint(0,len(windows) -1)])
		addWindow(desktop,outline,window)


	#add taskbar
	taskbar = cv.imread('.\\myassets\\taskbars\\' + taskbars[random.randint(0,len(taskbars)-1)])
	addTaskBar(desktop,outline,taskbar,desktopScreenPrimary)

	#add cursor	
	cursor_chosen = cursors[random.randint(0,len(cursors)-1)]
	cursor = cv.imread('.\\myassets\\cursors\\'+ cursor_chosen)
	c_w, c_h, c_c = cursor.shape
	cursor_pos = addCursor(desktop,cursor)
	#resize to the resolution of our video frames
	desktop = cv.resize(desktop,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
	outline = cv.resize(outline,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)

	return desktop, outline, cursor_pos

def generateDataBatch(samples=32, windows_generated=12, outline_channels=1):
	desktop, outline, cursor_pos = makeDesktop(windows_generated=windows_generated,outline_channels=outline_channels)
	
	desktop_height, desktop_width, desktop_channels = desktop.shape()
	desktop_batch = np.zeros((samples,desktop_height,desktop_width,desktop_channels))
	desktop_batch[0] = desktop

	outline_height, outline_width, outline_channels = outline.shape()
	outline_batch = np.zeros((samples,outline_height,outline_width,outline_channels))
	outline_batch[0] = outline

	cursor_pos_batch = np.zeros((samples,2))
	cursor_pos_batch[0] = list(cursor_pos)

	for i in range(1,samples):
		desktop, outline, cursor_pos = makeDesktop(windows_generated=windows_generated,outline_channels=outline_channels)
		desktop_batch[i] = desktop
		outline_batch[i] = outline
		cursor_pos_batch[i] = list(cursor_pos)

	return desktop_batch, outline_batch, cursor_pos_batch


key = 0
#for index, val in enumerate(sys.argv):	#first param index 1
args = sys.argv
argc = len(sys.argv)

extension = '.jpg'
image_name = 'desktop_image_'
outline_name = 'outline_image_'
together_name = 'deskline_'

#TODO: refactor to use the methods above
if __name__ == "__main__":
	#TODO: make a script that can take .txt parametrizing this and copy to clipboard
	max_generation =  int(args[1]) if argc > 1 else 9999
	windows_generated = int(args[2]) if argc > 2 else 8
	#individual - store outline and desktop individually
	#together - store outline and desktop together
	storage_mode =  args[3] if argc > 3 else 'individual'
	scaleFactor = float(args[4]) if argc > 4 else 1.0/2.5
	store = argc > 1
	show = not store

	image_num = 0
	while(key != 113 and key != 27 and image_num < max_generation):	#q to quit
		desktopScreens = 3
		desktopScreenPrimary = 2
		
		desktops = np.zeros((desktopScreens,1080,1920,3),dtype=np.uint8)
		desktop = np.zeros((1080,1920*desktopScreens,3),dtype=np.uint8)
		#outline would ideally have 1 channel, but I don't know how to 
		#inflate that in order to merge them for the together picture
		outline = np.zeros((1080,1920*desktopScreens),dtype=np.uint8)
		print(outline.shape)
		for i in range(0,desktopScreens):
			desktops[i] = cv.imread('.\\myassets\\wallpapers\\' + wallpapers[random.randint(0,len(wallpapers)-1)])
			desktop[0:1080,1920*i:1920*(i+1)] = desktops[i,:,:]
		
		if(key>=48 and key<=57):
			windows_generated = 1
			for i in range(0,key-48):
				windows_generated *= 2
		
		#start making windows
		for i in range(0,windows_generated):
			#print('adding window ' + str(i))
			window = cv.imread('.\\myassets\\windows\\' + windows[random.randint(0,len(windows) -1)])
			addWindow(desktop,outline,window)


		#add taskbar
		taskbar = cv.imread('.\\myassets\\taskbars\\' + taskbars[random.randint(0,len(taskbars)-1)])
		addTaskBar(desktop,outline,taskbar,desktopScreenPrimary)

		#add cursor	
		cursor_chosen = cursors[random.randint(0,len(cursors)-1)]
		cursor = cv.imread('.\\myassets\\cursors\\'+ cursor_chosen)
		c_w, c_h, c_c = cursor.shape
		x,y = addCursor(desktop,cursor)

		#if(key != 101):
		desktop = cv.resize(desktop,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
		outline = cv.resize(outline,(0,0),fx=scaleFactor,fy=scaleFactor,interpolation=cv.INTER_AREA)
		#else:
		#	desktop = cv.resize(desktop,(1600 * desktopScreens,900),interpolation = cv.INTER_AREA)
		#	outline = cv.resize(outline,(1600 * desktopScreens,900),interpolation = cv.INTER_AREA)
		#cv.imshow('Cursor: ' + cursor_chosen +'\tx : ' + s	tr(x) + '\ty : ' + str(y), desktop)
		#selector disappears in downsizing
		if show:
			cv.imshow('desktop',desktop)
			cv.imshow('outlines',outline)
			#print(desktop.shape)
			key = cv.waitKey(0)
			#print(key)
			cv.destroyAllWindows()
		if store:
			print('Storage mode: ' + storage_mode)
			if storage_mode == 'individual':
				cv.imwrite(image_name+str(image_num)+extension,desktop)
				cv.imwrite(outline_name+str(image_num)+extension,outline)
			if storage_mode == 'together':
				print('Storage mode together')
				h, w, c = desktop.shape
				deskline = np.zeros((h*2,w,c),dtype=np.uint8)
				print('Deskline shape: ' + str(deskline.shape))
				print('Desktop shape: ' + str(desktop.shape))
				print('Outline shape: ' + str(outline.shape))
				deskline[0:h,:] = desktop[:,:]
				deskline[h:,:] = outline[:,:]
				cv.imwrite(together_name + str(image_num) + extension, deskline)
		image_num += 1



