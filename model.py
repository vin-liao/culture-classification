import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.applications import vgg16, resnet50

from keras.layers import GlobalMaxPooling1D, Bidirectional, LSTM, Embedding

class Image_classification_models():
	def __init__(self, num_classes, image_size):
		self.num_classes = num_classes
		self.image_size = image_size

	def conv_model_1(self):
		model = Sequential()

		model.add(Conv2D(16, 3, input_shape=(self.image_size, self.image_size, 3)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(32, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(64, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Flatten())
		model.add(Dense(32))
		model.add(Dropout(0.5))
		model.add(Activation('relu'))

		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		model.name = 'Convolution_model_1'
		return model

	def conv_model_2(self):
		model = Sequential()
		model.add(Conv2D(16, 3, input_shape=(self.image_size, self.image_size, 3)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(32, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(64, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(128, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Conv2D(256, 3))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D())

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Dropout(0.5))
		model.add(Activation('relu'))

		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		model.name = 'Convolution_model_2'
		return model

	def vgg_transfer(self):
		model = vgg16.VGG16(include_top=False, input_shape=(self.image_size, self.image_size, 3))

		for layer in model.layers[:-4]:
			#freeze the layers
			layer.trainable = False

		x = model.output
		x = Flatten()(x)
		x = Dense(256, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(128, activation='relu')(x)
		pred = Dense(self.num_classes, activation='softmax')(x)

		model = Model(inputs=model.input, outputs=pred)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		model.name = 'VGG16_transfer'
		return model

	def resnet_transfer(self):
		model = resnet50.ResNet50(include_top=False, input_shape=(self.image_size, self.image_size, 3))

		for layer in model.layers:
			#freeze all layers
			layer.trainable = False

		x = model.output
		x = Flatten()(x)
		x = Dense(64, activation='relu')(x)
		pred = Dense(self.num_classes, activation='softmax')(x)

		model = Model(inputs=model.input, outputs=pred)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		model.name = 'Resnet_transfer'
		return model

	def get_model_list(self):
		return [self.conv_model_1(), self.conv_model_2(), self.vgg_transfer(), self.resnet_transfer()]

class Text_classification_models():
	def __init__(self, units, dropout, recurrent_dropout, num_classes):
		self.units = units
		self.dropout = dropout
		self.recurrent_dropout = recurrent_dropout
		self.num_classes = num_classes

	def bi_rnn(self):
		model = Sequential()

		#tuning the hyperparameter is required for good accuracy
		model.add(Bidirectional(LSTM(units=self.units,
			return_sequences=True,
			activation='relu',
			dropout=self.dropout,
			recurrent_dropout=self.recurrent_dropout),
            input_shape=(10, 100))) #10 sequence length, 100 dimension embedding

		#use max pool to select the highest result
		model.add(GlobalMaxPooling1D())
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		return model
