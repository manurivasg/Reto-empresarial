import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

class Model:
    def __init__(self):
      self.model = load_model('RiceClassifier.h5')

    def predict(self, file):
       img = cv2.imread(file)
       img = tf.image.resize(img, (64,64))
       prediction = self.model.predict(np.expand_dims(img/255, 0))
       return np.argmax(prediction)
