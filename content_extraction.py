import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_img, preprocess_input, get_intermediate_layers_model, clip_image_0_1, show_feature_maps
import sys
import getopt

'''
In this module, the content of a target image is reconstructed starting from a random noise image.
At every step, the content of the generated image is refined in order to minimize the loss (mean
squared distance) with the features extracted from the content target image.

Usage:  content_extraction.py [-v] -c <content image> [-l <layer name>] [-s <update steps]>
Options:    -v                  : visualization of the feature maps in the content output layer
            -c <content_image>  : path to the target image
            -l <layer name>     : name of the layer used to extract the features. Recommended: block4_conv2 (default), block4_conv1, block3_conv1
            -s <update steps>   : number of the update steps done to refine the generated image. Default: 10
'''


class ContentExtractor(tf.keras.models.Model):
    def __init__(self, content_layers):
        super(ContentExtractor, self).__init__()
        self.vgg = get_intermediate_layers_model(content_layers) # vgg is a Model(input, output), not the function vgg_layers
        self.content_layers = content_layers
        self.vgg.trainable = False

    def __call__(self, inputs):
        # "inputs" is an image with float values between and 1
        inputs = inputs * 255
        preprocessed_input = preprocess_input(inputs)
        content_outputs = self.vgg(preprocessed_input)  # content features of the image
        content_dict = {self.content_layers[0]: content_outputs}
        return content_dict


def content_loss(content_outputs, content_targets, weight=0.5):
    loss = weight * tf.add_n(
        [tf.reduce_mean((content_outputs[l] - content_targets[l])**2) for l in content_outputs.keys()])
    return loss


def update_image_step(gen_image, content_target, optimizer, extractor):
    with tf.GradientTape() as tape:
        gen_output = extractor(gen_image)
        loss = content_loss(gen_output, content_target)
    grad = tape.gradient(loss, gen_image)
    optimizer.apply_gradients([(grad, gen_image)])
    gen_image.assign(clip_image_0_1(gen_image))
    return loss


def main(argv):
    features_visualization = False
    image_path = "imgs/content/san_petronio.jpg"
    content_layers = ["block4_conv2"]
    steps = 10

    try:
        opts, args = getopt.getopt(argv, "hvc:l:s:", ["help", "visualization", "content=", "layer=", "steps="])
    except getopt.GetoptError:
        print('content_extraction.py [-v] -c <content image> [-l <layer name>] [-s <update steps>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('USAGE: content_extraction.py [-v] -c <content image> [-l <layer name>] [-s <update steps>]')
            sys.exit()
        elif opt in ("-v", "--visualization"):
            features_visualization = True
        elif opt in ("-c", "--content"):
            image_path = arg
        elif opt in ("-l", "--layer"):
            content_layers = [arg]
        elif opt in ("-s", "--steps"):
            steps = int(arg)

    content_image = load_img(image_path)    # tf.Tensor([[[0.69411767 0.7372549  0.8470589 ] [0.69803923 0.7411765  0.85098046] [0.7019608  0.74509805 0.854902 ] ...
                                            # shape (1, 450, 600, 3)
    extractor = ContentExtractor(content_layers)
    content_targets = extractor(content_image)    # <keras.engine.training.Model object at 0x63c558710>

    if features_visualization:
        show_feature_maps(extractor.vgg, content_image)

    white_noise = np.random.uniform(0, 1, content_image.shape)
    gen_image = tf.Variable(white_noise, dtype=tf.float32)    # shape (1, 450, 600, 3)
                                            # <tf.Variable 'Variable:0' shape=(1, 450, 600, 3) dtype=float64, numpy=
                                            # array([[[[8.50409808e-01, 7.47302637e-01, 6.86631100e-02],
                                            #          [3.78017295e-01, 1.05540438e-02, 2.99539570e-01],
                                            #          [7.64800506e-01, 5.53654046e-01, 2.33490189e-01], ... ,
    plt.imshow(np.array(gen_image[0]))
    plt.show()

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

    for _ in tqdm(range(steps), file=sys.stdout):
        if _ == 1:
            print("\tLoss at step 1:\t{}".format(loss))
        elif _ != 0 and _ % print_step == 0:
            print("\tLoss at step {}:\t{}".format(_, loss))
        loss = update_image_step(gen_image, content_targets, opt, extractor)
    print("\t"*12 + "Loss at step {}:\t{}".format(steps, loss))

    plt.imshow(np.array(gen_image[0]))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
