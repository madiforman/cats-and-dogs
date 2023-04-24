"""
=====================================================================================
Name: classify.py
Author: Madi Sanchez-Forman
Version: 5.23.23
Decription: This script takes a saved neurel network (.h5 file), and any number of images
to classify. It loads the neurel network given, and prints if it predicts each image is either
a cat or a dog.
=====================================================================================
"""
import sys
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils import img_to_array
import pandas as pd

#*********** Helper Functions ***********#
def prepare_image(filename):
    """
    Prepare image loads the image that is being predicted, and prepares it to go into the neurel network.
    It must be reshaped, and converted to have 3 channels. 
    params: filename of image
    returns: numpy array of converted image
    """
    img = load_img(filename, target_size=(100,100))
    img = img_to_array(img)
    img = img.reshape(1, 100, 100, 3)
    return img

#*********** Driver Function ***********#
def main():
    """
    Main method
    params: a compiled and trained model file ending in .h5 and at least one image
    """
    if len(sys.argv) < 3 or sys.argv[1].endswith('.hd5') == False: #ensure the file being given ends with .h5 and that images exist
        print("Usage: python3 classify.py <neurel_network.h5> <list of images to classify>")
        sys.exit()
    else:
        NN_NAME = sys.argv[1]
        model = load_model(NN_NAME) #load given model
        for img in sys.argv[2:]: #for each image after the model name
            prepped_img = prepare_image(img) #load it
            result = model.predict(prepped_img)[0] #make prediction
            if result == 0: #will output 0 for cats and 1 for dogs
                print("Image: " + img + " is (probably) a photo of a cat!") #print result of prediction
            elif result == 1:
                print("Image: " + img + " is (probably) a photo of a dog!")

if __name__ == "__main__":
    main()
