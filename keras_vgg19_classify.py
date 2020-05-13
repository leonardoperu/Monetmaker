import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import os
import utils


def print_predictions(labels, top):
    print("="*30)
    for i in range(len(labels)):
        print("Image {} - best {} predictions:".format(i, top))
        for score in labels[i]:
            print("[ {} , {:.2f}%% ]".format(score[1], score[2]*100))
        print("-"*30)
    print("="*30)


'''preparing the vgg-19 model'''
model = VGG19()
print(model.summary())
print("LAYERS:\n"+str(model.layers))
# plot_model(model, to_file=os.getcwd()+'/imgs/generated/model/vgg19.png')


'''preprocessing the input image'''
img1 = utils.prepare_img('imgs/vgg19_classification/teamug.png')
img2 = utils.prepare_img('imgs/content/san_petronio.jpg')

'''predicting'''
pred = model.predict(img1)
labels = decode_predictions(pred, top=3)
print()
print_predictions(labels, 3)


'''show feature maps'''
layer_names = ["block1_conv1",
               "block2_conv2",
               "block3_conv4",
               "block4_conv4",
               "block5_conv4"]
utils.show_feature_maps(model, layer_names, img2)
