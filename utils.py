import numpy as np
import PIL
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot


def prepare_img(img_path):
    # load the image with the required shape
    img = load_img(img_path, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    # reshaping in 4 dimensions: samples, rows, columns, channels
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    # prepare the image (e.g. scale pixel values for the vgg)
    # this subtracts the avg RGB value, as specified in the paper https://arxiv.org/abs/1409.1556
    img = preprocess_input(img)
    return img


def load_img(path, max_dim=512):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)  # f.Tensor([[[177 188 216] [178 189 217] [179 190 218] ...
    img = tf.image.convert_image_dtype(img, tf.float32) # tf.Tensor([[[0.69411767 0.7372549  0.8470589 ] [0.69803923 0.7411765  0.85098046] [0.7019608  0.74509805 0.854902 ] ...
                                                        # shape: (450, 600, 3)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    longest_dim = max(shape)
    scale = max_dim / longest_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]    # shape (1, 450, 600, 3)
    return img


def clip_image_0_1(img):
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor*255.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_layers_by_name(model, layer_names):
    return [model.get_layer(l) for l in layer_names]


def get_outputs_by_layer_names(model, layer_names):
    result = [model.get_layer(l).output for l in layer_names]
    return result


def get_intermediate_layers_model(layer_names):
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg19.trainable = False
    output_layers = get_outputs_by_layer_names(vgg19, layer_names)
    intermediate_layers_model = tf.keras.Model([vgg19.input], output_layers)
    return intermediate_layers_model


def show_feature_maps(model, image):
    feature_maps = model.predict(image)
    # plot the output from each block
    square = 8
    for fmap in feature_maps:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[:, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.show()
