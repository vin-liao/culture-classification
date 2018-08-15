import model
import cv2
import os
import numpy as np
import utils
import embedding_utils

#hyperparameter
#PS: hm_epoch means how many epoch
hm_epoch = 20
hm_batch = 32
image_size = 224

model_path = './data/models'

def train_text():
	x_train, x_valid, x_test, y_train, y_valid, y_test = utils.get_text_data()
	embedding = embedding_utils.Embedding()

	rnn_units = 128
	dropout = 0.2
	recurrent_dropout = 0.2
	num_classes = len(utils.text_label_list)

	models = model.Text_classification_models(units=rnn_units,
		dropout=dropout,
		recurrent_dropout=recurrent_dropout,
		num_classes=num_classes)

	rnn_model = models.bi_rnn()
	rnn_model.fit(x_train, y_train, epochs=hm_epoch, batch_size=hm_batch)
	rnn_model.evaluate(x_test, y_test, batch_size=hm_batch, verbose=0)

	print('Saving model...')
	rnn_model.save(model_path + '/rnn_model.h5')

def train_image():
	x_train, x_valid, x_test, y_train, y_valid, y_test = utils.get_image_data(image_size)

	#instantiate the class
	models = model.Image_classification_models(num_classes=len(utils.image_label_list), \
		image_size=image_size)

	model_list = models.get_model_list()
	trained_model = []

	#train all the models in model_list
	for one_model in model_list:
		one_model.fit(x_train, y_train, epochs=hm_epoch, batch_size=hm_batch)
		trained_model.append(one_model)

	#evluate the trained models
	best_acc = 0
	for one_model in trained_model:
		train_metric_result = one_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)
		validation_metric_result = one_model.evaluate(x_valid, y_valid, batch_size=hm_batch, verbose=0)

		print('Model name:', one_model.name)
		print('Train Accuracy:', train_metric_result[1])
		print('Validation Accuracy:', validation_metric_result[1])
		print('\n')

		#pick the best model based on their accuracy
		if validation_metric_result[1] > best_acc:
			best_acc = validation_metric_result[1]
			best_model = one_model

	print('The best model is: %s with %.2f %% accuracy' %(best_model.name, best_acc))

	#test the model on test set
	best_model_train = best_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)
	best_model_test = best_model.evaluate(x_test, y_test, batch_size=hm_batch, verbose=0)

	print('Train loss: %.3f acc %.3f' %(best_model_train[0], best_model_train[1]))
	print('Test loss: %.3f acc %.3f' %(best_model_test[0], best_model_test[1]))

	#save the best model
	print('Saving model...')
	best_model.save(model_path + '/best_image_model.h5')

def train_spectogram():
    x_train, x_valid, x_test, y_train, y_valid, y_test = utils.get_spectogram_data(image_size)

    #instantiate the class
    models = model.Image_classification_models(num_classes=len(utils.spectogram_label_list), \
            image_size=image_size)

    model_list = models.get_model_list()
    trained_model = []

    #train all the models in model_list
    for one_model in model_list:
        one_model.fit(x_train, y_train, epochs=hm_epoch, batch_size=hm_batch)
        trained_model.append(one_model)

    #evluate the trained models
    best_acc = 0
    for one_model in trained_model:
        train_metric_result = one_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)

        validation_metric_result = one_model.evaluate(x_valid, y_valid, batch_size=hm_batch, verbose=0)

        print('Model name:', one_model.name)
        print('Train Accuracy:', train_metric_result[1])
        print('Validation Accuracy:', validation_metric_result[1])
        print('\n')

        #pick the best model based on their accuracy
        if validation_metric_result[1] > best_acc:
            best_acc = validation_metric_result[1]
            best_model = one_model

    print('The best model is: %s with %.2f %% accuracy' %(best_model.name, best_acc))

    #test the model on test set
    best_model_train = best_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)
    best_model_test = best_model.evaluate(x_test, y_test, batch_size=hm_batch, verbose=0)

    print('Train loss: %.3f acc %.3f' %(best_model_train[0], best_model_train[1]))
    print('Test loss: %.3f acc %.3f' %(best_model_test[0], best_model_test[1]))

    #save the best model
    print('Saving model...')
    best_model.save(model_path + '/best_spectogram_model.h5')

if __name__ == "__main__":
    train_text()
    train_spectogram()
    train_image()
