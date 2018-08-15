import cv2
import os
import numpy as np
import pathlib
from itertools import chain
from sklearn.model_selection import train_test_split

#text
import re
from nltk import word_tokenize
import gensim
import embedding_utils

image_label_list = [] #list of category names to avoid confusion
text_label_list = [] #list of category names to avoid confusion
spectogram_label_list = []

def get_spectogram_data(image_size=224):
    number_of_image_parts = 10
    spectogram_path = './data/spectrogram'
    total_image_file = 0

    for root, dirs, files in os.walk(spectogram_path):
        for file in files:
            total_image_file+=1
    one_image = cv2.imread(root + '/' + file)
    img_height, img_width = one_image.shape[:2]

    target_list = []
    #x is a np array with shape (height, height, 3), because the aspect ratio is kept the same
    x = np.zeros(shape=(total_image_file*number_of_image_parts, \
            image_size, image_size, 3), dtype=np.uint8)

    category = os.listdir(spectogram_path)
    for i, cat in enumerate(category):
        spec_name_list = os.listdir(spectogram_path + '/{}'.format(cat))
        spectogram_label_list.append(cat)
        for spectogram_name in spec_name_list:
            spectogram_full_path = '%s/%s/%s' %(spectogram_path, cat, spectogram_name)

            image = cv2.imread(spectogram_full_path)
            for j in range(number_of_image_parts):
                hm_width = img_width//10 #how much pixel width per part
                start_pixel = j*hm_width
                end_pixel = (j+1)*hm_width
                #crop = im[y1:y2, x1:x2]
                #(x1, y1) = top, left; (x2, y2) = bottom right
                cropped_image = image[:, start_pixel:end_pixel]

                resized_image = cv2.resize(cropped_image, (image_size, image_size))
                x[i] = resized_image

                #create the categorical target list
                #e.g. Batak: 1, Betawi: 2, Toraja: 3, ...
                target_list.append(i+1)

    y = np.zeros((len(target_list), len(spectogram_label_list)), dtype=np.int32)
    #===turn categorical target into one hot===
    for i, target in enumerate(target_list):
        #from the zero array, set the value of the corresponding index to 1
        y[i][target-1] = 1

    #===splitting data===
    #train/valid/test = 70/15/15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def get_image_data(image_size=224):
    #todo: instead of appending the data into lists, just create np zeros and fill it.
    image_path = './data/images'
    total_image_file = 0
    count = 0

    #count the total image file in the /data/image folder
    for root, dirs, files in os.walk(image_path):
        for file in files:
            total_image_file+=1

    x = np.zeros((total_image_file, image_size, image_size, 3), dtype=np.uint8)
    target_list = []

    #===reading images into image_list array===
    category = os.listdir(image_path)
    for i, cat in enumerate(category):
        img_list = os.listdir(image_path + '/{}'.format(cat))
        image_label_list.append(cat)
        for image_name in img_list:
            #insert the image into np array
            x[count, :] = cv2.resize(cv2.imread('%s/%s/%s' %(image_path, cat, image_name)),\
                    (image_size, image_size))
            count+=1

            #create the categorical target list
            #e.g. Batak: 1, Betawi: 2, Toraja: 3, ...
            target_list.append(i+1)

    #===create one hot vector===
    y = np.zeros((total_image_file, len(image_label_list)), dtype=np.int32)
    for i, target in enumerate(target_list):
        #from the zero array, set the value of the corresponding index to 1
        y[i][target-1] = 1

    #===splitting data===
    #train/valid/test = 70/15/15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def get_text_data():
    max_sequence_length = 10 #maximum sequence length of RNN
    dim_size = 100 #dimension of embedding

    text_path = './data/text'

    sentence_list = []
    target_list = []

    #===read text into text_list array===
    category = os.listdir(text_path)
    for i, cat in enumerate(category):
        txt_list = os.listdir(text_path + '/{}'.format(cat))
        text_label_list.append(cat)
        for text_name in txt_list:
            text_full_dir = '%s/%s/%s' %(text_path, cat, text_name)
            with open(text_full_dir, 'r') as f:
                #preprocess: splitting the text into list, separated by \n
                texts = f.readlines()
                for text in texts:
                    sentence = re.sub('\n', '', text)
                    tokens = word_tokenize(sentence)

                    #append the sentence into a list
                    sentence_list.append(tokens)

                    #create the categorical target list
                    #e.g. Batak: 1, Betawi: 2, Toraja: 3, ...
                    target_list.append(i+1)

    #===get embedding model===
    #instantiate the class
    embedding = embedding_utils.Embedding(sentence_list)
    model = embedding.get_embedding_model()

    #===representing words with word vectors===
    x = np.zeros((len(sentence_list), max_sequence_length, dim_size))
    y = np.zeros((len(sentence_list), len(text_label_list)), dtype=np.int32)

    """
    fill the vectors into the np array, if the sentence is longer than the maximum
    sequence length, index error will be raised, and ignored (pass). If the sentence is more
    than max seq length, then only the first len(max seq len) words are turned into vectors
    """
    for index, sentence in enumerate(sentence_list):
        try:
            for token_index, token in enumerate(sentence):
                x[index, token_index, :] = model[token]
        except:
            pass

    #===turn categorical target into one hot===
    for i, target in enumerate(target_list):
        #from the zero array, set the value of the corresponding index to 1
        y[i][target-1] = 1

    #===splitting data===
    #train/valid/test = 70/15/15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, x_valid, x_test, y_train, y_valid, y_test
