import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_img, preprocess_input, get_intermediate_layers_model, clip_image_0_1, show_feature_maps
import sys
import getopt
from PIL import Image

'''
In this module, the style of a target image is reconstructed over a base image with
its own content. At every step, the artistic style of the generated image is refined
in order to minimize the loss with the features extracted from the style target image.

Usage:  style_extraction.py [-v] -t <style image> -b <base image> [-s <update steps>]
Options:    -v                  : visualization of the feature maps in the content output layer
            -t <style_image>    : path to the target image
            -b <base_image>     : base for the generated image
            -s <update steps>   : number of the update steps done to refine the generated image. Default: 10
'''


def gram(x):
    """
    Return a gram matrix for the given input matrix.
    Args:
        x: the matrix to calculate the gram matrix of
    Returns:
        the gram matrix of x
    """
    shape = K.shape(x[0])
    # flatten the 3D tensor by converting each filter's 2D matrix of points
    # to a vector, thus we have the matrix: [filter_width x filter_height, num_filters]
    F = K.reshape(x, (shape[0] * shape[1], shape[2]))
    # take inner product over all the vectors to produce the Gram matrix
    product = K.dot(K.transpose(F), F)
    product /= tf.cast(shape[0]*shape[1], tf.float32)
    shape = K.shape([product])
    return K.reshape(product, shape)


class StyleExtractor(tf.keras.models.Model):
    """
    An instance of this class, when used to process an input image, returns
    its style features (Gram matrix) extracted from the specified layers.
    In this application, the chosen layers are: "block1_conv1", "block2_conv1",
    "block3_conv1", "block4_conv1", "block5_conv1".
    """
    def __init__(self, style_layers):
        super(StyleExtractor, self).__init__()
        self.vgg = get_intermediate_layers_model(style_layers) # vgg is a Model(input, output), not the function vgg_layers
        self.style_layers = style_layers
        self.vgg.trainable = False

    def __call__(self, inputs):
        # "inputs" is an image with float values between and 1
        inputs = inputs * 255
        preprocessed_input = preprocess_input(inputs)
        style_outputs = self.vgg(preprocessed_input)  # style features of the image
        style_outputs = [gram(s) for s in style_outputs]
        style_dict = {layer: out for layer, out in zip(self.style_layers, style_outputs)}
        return style_dict


def style_loss(style_outputs, style_targets, weight=1.0):
    """
    Loss function to compute the distance between the style representation (Gram matrix)
    of the processed image (style_outputs) and of the target style image (style_targets)
    """
    loss = weight * tf.add_n(
        [tf.reduce_mean((style_outputs[l] - style_targets[l]) ** 2) for l in style_outputs.keys()])
    loss /= float(len(style_outputs))
    return loss


def update_image_step(gen_image, style_target, optimizer, extractor):
    with tf.GradientTape() as tape:
        gen_output = extractor(gen_image)
        loss = style_loss(gen_output, style_target)
    grad = tape.gradient(loss, gen_image)
    optimizer.apply_gradients([(grad, gen_image)])
    gen_image.assign(clip_image_0_1(gen_image))
    return loss


def main(argv):
    features_visualization = False
    image_path = "imgs/style/starry_night.jpeg"
    gen_img_base = "imgs/content/san_luca.jpg"
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

    steps = 1

    try:
        opts, args = getopt.getopt(argv, "hvt:b:s:", ["help", "visualization", "style=", "base=", "steps="])
    except getopt.GetoptError:
        print('style_extraction.py [-v] -t <style image> -b <base image> [-s <update steps>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('USAGE: style_extraction.py [-v] -t <style image> -b <base image> [-s <update steps>]')
            sys.exit()
        elif opt in ("-v", "--visualization"):
            features_visualization = True
        elif opt in ("-t", "--style"):
            image_path = arg
        elif opt in ("-b", "--base"):
            gen_img_base = arg
        elif opt in ("-l", "--layer"):
            style_layers = [arg]
        elif opt in ("-s", "--steps"):
            steps = int(arg)

    style_image = load_img(image_path)    # tf.Tensor([[[[0.7294118  0.59607846 0.45882356] [0.7568628  0.62352943 0.48627454] [0.6509804  0.52156866 0.3921569] ...
                                            # shape (1, 324, 512, 3)
    #white_noise = np.random.uniform(0, 1, style_image.shape)
    gen_image = load_img(gen_img_base)
    gen_image = tf.Variable(gen_image, dtype=tf.float32)    # shape (1, 324, 512, 3)
                                            # <tf.Variable 'Variable:0' shape=(1, 324, 512, 3) dtype=float64, numpy=
                                            # array([[[[8.50409808e-01, 7.47302637e-01, 6.86631100e-02],
                                            #          [3.78017295e-01, 1.05540438e-02, 2.99539570e-01],
                                            #          [7.64800506e-01, 5.53654046e-01, 2.33490189e-01], ... ,
    #plt.imshow(np.array(gen_image[0]))
    #plt.show()

    extractor = StyleExtractor(style_layers)
    style_targets = extractor(style_image)

    if features_visualization:
        show_feature_maps(extractor.vgg, style_image)

    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    print()

    print_step = 1
    if steps <= 15:
        pass
    elif steps <= 40:
        print_step = 4
    elif steps <= 100:
        print_step = 10
    else:
        print_step = 20

    loss = 0
    save_as = "imgs/generated/style/" + image_path.split("/")[-1].split(".")[0] + "_"
    save_as += gen_img_base.split("/")[-1].split(".")[0] + "_" + str(steps) + "_steps.jpg"

    for _ in tqdm(range(steps), file=sys.stdout):
        if _ == 1:
            print("\tLoss at step 1:\t{}".format(loss))
        elif _ != 0 and _ % print_step == 0:
            print("\tLoss at step {}:\t{}".format(_, loss))
        loss = update_image_step(gen_image, style_targets, opt, extractor)
    print("\t"*12 + "Loss at step {}:\t{}".format(steps, loss))



    plt.imshow(np.array(gen_image[0]))
    plt.show()
    array = np.array((gen_image[0] * 255), dtype=np.uint8)
    Image.fromarray(array).save(save_as)


if __name__ == "__main__":
    main(sys.argv[1:])
