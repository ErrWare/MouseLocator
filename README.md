# MouseLocator
Used to roughly locate a mouse cursor inside of a picture
Requires openCV2

# Files
clickToFind.py
	opens images, click on the cursor to crop & save it & remove image from dataset

cursorClassifier.py
	sets up and trains VGG-16 based classifiers. Saves architecture and weight files

interactiveFinder.py
	uses a specified classifier architecture to generate proposals for regions w/ cursor from an image. User inputs whether classifier was right or not.

croppedCursorGen / cursorGen.py
	Programmatically generate picture with cursors located in them. Will take some tweaking to make it right. I think I hardcoded some file paths and such that should be changes.

myAssests
	some of the files used for cursorGeneration