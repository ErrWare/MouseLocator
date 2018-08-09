#Cursor classifier for

from keras.models import Model, model_from_json
from keras.layers import Dense, Flatten
import keras
from keras import optimizers
import croppedCursorGen as dataGenerator
from matplotlib import pyplot as plt
import cv2 as cv
import json

#Left to test:
	#Dropout
	#Batch norm?
BATCH_SIZE = 96
EPOCHS = 40	#before finetune
EPOCH_STEP = 1
#SGD parameters copied from dogs-cats keras tut
#this sgd seems poor for initial training
OPTIMIZERS = [('rms',optimizers.rmsprop())]#,('sgd',optimizers.SGD(lr=1e-4, momentum=0.9))]
#EPOCHS to spend finetuning the last N vgg layers
FT_N_EPOCHS = [[3,3,6]]
FC_SIZE = [[32],[64],[128],[32,32],[64,32]]
LAST_TRAINABLE_VGG_LAYER = 18
BINARY = False
PREV_WEIGHT_FILE = ''

for optimizer in OPTIMIZERS:
	for FC_layer_config in FC_SIZE:
		architecture_name = ('' if BINARY else 'CAT_') +'FC'
		vgg = keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(48,48,3))
		vgg_input = vgg.inputs
		vgg_output = vgg.outputs

		#freeze the vgg layers
		for index, layer in enumerate(vgg.layers):
			layer.trainable = False

		#flatten vgg output tensor
		model_tensor = Flatten()(vgg_output[0])

		#add the FC layers
		for FC_layer_size in FC_layer_config:
			architecture_name = architecture_name + '-' +str(FC_layer_size)
			model_tensor = Dense(FC_layer_size, activation='relu')(model_tensor)

		model_tensor = Dense(2 if BINARY else dataGenerator.CATEGORIES, activation='softmax')(model_tensor)


		model = Model(inputs=vgg_input,outputs=model_tensor)
		print('Model architecture made')
		print(model.summary())
		#CHOSEN ARBITRARILY FOR NOW
		architecture_name = architecture_name + '-' + optimizer[0]
		model.compile(optimizer=optimizer[1],
						loss='binary_crossentropy' if BINARY else 'categorical_crossentropy',
						metrics=['accuracy'])
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('Model Compiled: ' + architecture_name)			
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

		#Straight outa stack overflow	
		with open('ARCH_'+architecture_name+'.json', 'w') as outfile:
			json.dump(model.to_json(), outfile)
		
		#train top model
		for i in range(int(EPOCHS / EPOCH_STEP)):
			val_batch, val_labels = dataGenerator.generateDataBatch(512,binary=BINARY)
			batch, labels = dataGenerator.generateDataBatch(4096,binary=BINARY)
			model.fit(x=batch,y=labels,batch_size=BATCH_SIZE,epochs=EPOCH_STEP*(i+1),initial_epoch=EPOCH_STEP*i,verbose=1,validation_data=(val_batch,val_labels),shuffle=True)

		model.save_weights('WEIGHTS_'+architecture_name+'_afterInitial.h5')
		PREV_WEIGHT_FILE = 'WEIGHTS_'+architecture_name+'_afterInitial.h5'


#Decided to decouple initial training from finetuning
#Fine tune the last three layers of each architecture
optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
for FC_layer_config in FC_SIZE:
	#Construct architecture name
	architecture_name = 'FC'
	for FC_layer_size in FC_layer_config:
		architecture_name = architecture_name + '-' +str(FC_layer_size)
	architecture_name = architecture_name + '-' + optimizer[0]
	#Load architecture
	model = model_from_json('ARCH_'+architecture_name+'.json')
	model.load_weights('WEIGHTS_'+architecture_name+'_afterInitial.h5')
	#Do the fine tuning
	for FT_N_EPOCHS_CONFIGURATION in FT_N_EPOCHS:	#CONFIG Like [3,3,6]
		#Construct FT Config name
		architecture_name_FT = architecture_name + '_FT'
		for FT_EPOCHS in FT_N_EPOCHS:
			architecture_name_FT = architecture_name_FT + '-' + str(FT_EPOCHS)
		for index, FT_EPOCHS in enumerate(FT_N_EPOCHS_CONFIGURATION):
			for layer in vgg.layers[LAST_TRAINABLE_VGG_LAYER-index:]:
				layer.trainable = True;
			#compile and fine tune
			model.compile(optimizer=optimizer,
					loss='binary_crossentropy',
					metrics=['accuracy'])
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print('Model Compiled: ' + architecture_name)			
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			#Got following behavior after first fine tune of first model:
			#accuracy became 0.5000 immediately
			#Want to make sure it's not because the weights are reset after the compile
			model.load_weights(PREV_WEIGHT_FILE)
			print(model.summary())
			for epoch in range(FT_EPOCHS):
				val_batch, val_labels = dataGenerator.generateDataBatch(512, binary=BINARY)
				batch, labels = dataGenerator.generateDataBatch(4096, binary=BINARY)
				model.fit(x=batch,y=labels,batch_size=BATCH_SIZE,epochs=1,initial_epoch=0,verbose=1,validation_data=(val_batch,val_labels),shuffle=True)
			model.save_weights('WEIGHTS_'+architecture_name_FT+'_afterFT'+str(index)+'.h5')
			PREV_WEIGHT_FILE = 'WEIGHTS_'+architecture_name_FT+'_afterFT'+str(index)+'.h5'

print('Success')