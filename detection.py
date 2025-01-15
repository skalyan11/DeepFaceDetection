import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
image_generator = images.batch(3).as_numpy_iterator()

#while True:
    #plot_images = image_generator.next()
    #fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
    #for idx, image in enumerate(plot_images):
        #ax[idx].imshow(image)
    #plt.show()
    #input("Press Enter to load the next batch...")

