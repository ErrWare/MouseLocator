# MouseLocator
Used to roughly locate a mouse cursor inside of a picture
Requires openCV2

The idea was to train a classifier on programmatically generated examples of mouse cursors on backgrounds. Then use the classifier as a sliding window in the real dataset to find the cursor, hoping that it would transfer at least well enough to find the cursor in the first 10 or 20 or 30 (just not 200+) tries and allow me to build a sub dataset from the real dataset. clickToFind was my solution when that failed, and was what I was hoping to avoid.

Sliding window is old and definitely not a choice approach. I tried before this to use Retinanet and MRCNN projects to my ends. I couldn't figure those out. Hence the brutish homemade approach.

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
