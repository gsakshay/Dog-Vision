import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path, IMG_SIZE=229):
  """
  Input: Image file path, required image_size
  Output: Numeric represent of the image of type tensor (Normalised value)
  """
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image

def predict_dog_breed(folder_path, model):
    """
    Input: folder path containing images of dogs and model using which the prediction of breed has to be made
    Output: Pictorial representation of image with the label as predicted breed of the dog
    """


    file_names = [folder_path + fname for fname in os.listdir(folder_path)]
    data = tf.data.Dataset.from_tensor_slices((tf.constant(file_names)))
    data_batch = data.map(preprocess_image).batch(32)

    labels = pd.read_csv('labels.csv')
    breeds = np.unique(np.array(labels['breed']))

    preds = model.predict(data_batch, verbose=1)
    pred_labels = [breeds[np.argmax(preds[i])] for i in range(len(preds))]

    images = []
    for image in data_batch.unbatch().as_numpy_iterator():
      images.append(image)

    for i in range(len(file_names)):
      plt.figure(figsize=(3, 3))
      plt.imshow(images[i])
      plt.title(pred_labels[i])
      # Turn the grid lines off
      plt.axis("off")


INCEPTION_RESENT = tf.keras.models.load_model('models/INCEPTION_RESENT_V2-Adam-L2-ver2.h5', custom_objects={"KerasLayer":hub.KerasLayer})

def vision(path):
  return predict_dog_breed(path, INCEPTION_RESENT)
