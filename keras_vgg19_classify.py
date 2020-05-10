import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import os


'''preparing the vgg-19 model'''
model = VGG19()
print(model.summary())
#plot_model(model, to_file=os.getcwd()+'/imgs/generated/model/vgg19.png')


'''preprocessing the input image'''
img = load_img('imgs/vgg19_classification/cat.png', target_size=(224, 224))
img = img_to_array(img)
# reshaping in 4 dimensions: samples, rows, columns, channels
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
print(img.shape)
# this subtracts the avg RGB value, as specified in the paper https://arxiv.org/abs/1409.1556
img = preprocess_input(img)


'''predicting'''
pred = model.predict(img)
labels = decode_predictions(pred, top=3)
print("Best 3 predictions:\n"+str(labels))
