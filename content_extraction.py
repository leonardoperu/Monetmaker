import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_img, preprocess_input, get_intermediate_layers_model, clip_image_0_1

'''
In this module, the content of a target image is reconstructed starting from a random noise image.
At every step, the content of the generated image is refined in order to minimize the loss (mean
squared distance) with the features extracted from the content target image.
'''


class ContentExtractor(tf.keras.models.Model):
    def __init__(self, content_layers):
        super(ContentExtractor, self).__init__()
        self.vgg = get_intermediate_layers_model(content_layers) # vgg is a Model(input, output), not the function vgg_layers
        self.content_layers = content_layers
        self.vgg.trainable = False

    def call(self, inputs):
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


def update_image_step(gen_image, content_target, optimizer):
    with tf.GradientTape() as tape:
        gen_output = extractor(gen_image)
        loss = content_loss(gen_output, content_target)
    grad = tape.gradient(loss, gen_image)
    optimizer.apply_gradients([(grad, gen_image)])
    gen_image.assign(clip_image_0_1(gen_image))
    return loss


content_layers = ["block4_conv2"]

image_path = "imgs/content/san_petronio.jpg"
content_image = load_img(image_path)    # tf.Tensor([[[0.69411767 0.7372549  0.8470589 ] [0.69803923 0.7411765  0.85098046] [0.7019608  0.74509805 0.854902 ] ...
                                        # shape (1, 450, 600, 3)
extractor = ContentExtractor(content_layers)
content_targets = extractor(content_image)    # <keras.engine.training.Model object at 0x63c558710>

white_noise = np.random.uniform(0, 1, content_image.shape)
gen_image = tf.Variable(white_noise, dtype=tf.float32)    # shape (1, 450, 600, 3)
                                        # <tf.Variable 'Variable:0' shape=(1, 450, 600, 3) dtype=float64, numpy=
                                        # array([[[[8.50409808e-01, 7.47302637e-01, 6.86631100e-02],
                                        #          [3.78017295e-01, 1.05540438e-02, 2.99539570e-01],
                                        #          [7.64800506e-01, 5.53654046e-01, 2.33490189e-01], ... ,
plt.imshow(np.array(gen_image[0]))
plt.show()
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
loss = 0
print()
for _ in tqdm(range(1000)):
    loss = update_image_step(gen_image, content_targets, opt)
    if _ % 20 == 0:
        print("Loss at step {}: {}".format(_, loss))
print("Loss at step 999: {}".format(loss))
plt.imshow(np.array(gen_image[0]))
plt.show()

