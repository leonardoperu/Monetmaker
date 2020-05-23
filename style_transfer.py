import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_img, preprocess_input, clip_image_0_1, show_feature_maps
from content_extraction import content_loss, ContentExtractor
from style_extraction import style_loss, StyleExtractor, gram
import sys
import getopt
from PIL import Image


WEIGHTS = (8e-1, 4e1) # (style, content)


class StyleContentExtractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentExtractor, self).__init__()
        self.style_extractor = StyleExtractor(style_layers)
        self.content_extractor = ContentExtractor(content_layers)

    def __call__(self, inputs):
        # "inputs" is an image with float values between and 1
        inputs = inputs * 255
        preprocessed_input = preprocess_input(inputs)
        style_outputs = self.style_extractor.vgg(preprocessed_input)  # style features of the image
        style_outputs = [gram(s) for s in style_outputs]
        style_dict = {layer: out for layer, out in zip(self.style_extractor.style_layers, style_outputs)}
        content_outputs = self.content_extractor.vgg(preprocessed_input)
        content_dict = {self.content_extractor.content_layers[0]: content_outputs}
        return style_dict, content_dict


def total_loss(style_outputs, style_targets, content_outputs, content_targets, weights):
    s_loss = style_loss(style_outputs, style_targets, weights[0])
    c_loss = content_loss(content_outputs, content_targets, weights[1])
    loss = s_loss + c_loss
    return loss, s_loss, c_loss


def update_image_step(gen_image, style_target, content_target, optimizer, extractor, weights):
    with tf.GradientTape() as tape:
        gen_style_output, gen_content_output = extractor(gen_image)
        loss, s_loss, c_loss = total_loss(gen_style_output, style_target, gen_content_output, content_target, weights)
        loss += 40 * tf.image.total_variation(gen_image)
    grad = tape.gradient(loss, gen_image)
    optimizer.apply_gradients([(grad, gen_image)])
    gen_image.assign(clip_image_0_1(gen_image))
    return loss, s_loss, c_loss


def main(argv):
    features_visualization = False
    style_path = "imgs/style/starry_night.jpeg"
    content_path = "imgs/content/san_petronio.jpg"
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    content_layers = ["block4_conv2"]

    W = WEIGHTS
    steps = 2000

    try:
        opts, args = getopt.getopt(argv, "hvt:c:s:", ["help", "visualization", "style=", "content=", "steps="])
    except getopt.GetoptError:
        print('style_extraction.py [-v] -t <style image> -c <content image> [-s <update steps>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('USAGE: style_extraction.py [-v] -t <style image> -c <content image> [-s <update steps>]')
            sys.exit()
        elif opt in ("-v", "--visualization"):
            features_visualization = True
        elif opt in ("-t", "--style"):
            style_path = arg
        elif opt in ("-c", "--content"):
            content_path = arg
        elif opt in ("-l", "--layer"):
            style_layers = [arg]
        elif opt in ("-s", "--steps"):
            steps = int(arg)

    content_image = load_img(content_path)
    style_image = load_img(style_path)    # tf.Tensor([[[[0.7294118  0.59607846 0.45882356] [0.7568628  0.62352943 0.48627454] [0.6509804  0.52156866 0.3921569] ...
                                            # shape (1, 324, 512, 3)
    gen_image = tf.Variable(content_image, dtype=tf.float32)
    # white_noise = np.random.uniform(0, 1, content_image.shape)
    # gen_image = tf.Variable(white_noise, dtype=tf.float32)
    # plt.imshow(np.array(gen_image[0]))
    # plt.show()

    extractor = StyleContentExtractor(style_layers, content_layers)
    style_targets, _ = extractor(style_image)
    _, content_targets = extractor(content_image)

    if features_visualization:
        show_feature_maps(extractor.content_extractor.vgg, content_image)

    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    print()

    print_step = 1
    if steps <= 15:
        pass
    elif steps <= 40:
        print_step = 4
    elif steps <= 100:
        print_step = 10
    elif steps <= 500:
        print_step = 20
    else:
        print_step = 100

    loss, s_loss, c_loss = 0, 0, 0
    save_as = "imgs/generated/transfer/" + style_path.split("/")[-1].split(".")[0] + "_"
    save_as += content_path.split("/")[-1].split(".")[0] + "_" + str(steps) + "_steps.jpg"
    half = {500: (0.8, 40), 1000: (0.9, 40), 1500: (1, 40), 2000: (1, 40)}

    for _ in tqdm(range(steps), file=sys.stdout):
        if _ == 200:
            opt.learning_rate = 0.04
        if _ in half.keys():
            opt.learning_rate.assign(opt.learning_rate / 2)
            W = half[_]
            print("=" * 20 + "new weights: " + str(half[_]) + ", lr halved" + "=" * 20)
        if _ == 1 or _ != 0 and _ % print_step == 0:
            print("\tLoss at step {}:\t{}\t[ S: {:.2f}; C: {:.2f} ]".format(_, loss, s_loss, c_loss))
            plt.imshow(np.array(gen_image[0]))
            plt.show()
        loss, s_loss, c_loss = update_image_step(gen_image, style_targets, content_targets, opt, extractor, W)

    print("\t"*12 + "Loss at step {}:\t{}\t[ S: {:.2f}; C: {:.2f} ]".format(steps, loss, s_loss, c_loss))
    plt.imshow(np.array(gen_image[0]))
    plt.show()

    array = np.array((gen_image[0] * 255), dtype=np.uint8)
    Image.fromarray(array).save(save_as)


if __name__ == "__main__":
    main(sys.argv[1:])
