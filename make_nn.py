"""
=====================================================================================
Name: make_nn.py
Author: Madi Sanchez-Forman
Version: 5.23.23
Decription: This script takes a directory containing pictures of cats and dogs, and then
a file name ending in .h5 and builds a CNN to classify cats and dogs with 93% accuracy
=====================================================================================
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import sys
import os

#*********** Helper Functions ***********#
def process_data(path):
    """
    Process data takes a path to a list of dog and cat images, and builds a pandas dataframe of the images
    It then uses train_test_split from sklearn to split the data into a validation and testing set. It returns the
    a tuple (testing dataframe, validation dataframe)
    params: path -> path to dir of images
    returns: training dataframe, validation dataframe
    """
    dataset = [] #data set will be a list of dictionaries mapping filename -> dog or cat
    for f in os.listdir(path): #for each image file in the path
        if 'c' in f: #check if its cat or dog and add to dataframe
            dataset.append({'filename': f, 'label': 'c'})
        elif 'd' in f:
            dataset.append({'filename': f, 'label': 'd'})
    df = DataFrame(dataset) #create dataframe 
    train_df, validate_df = train_test_split(df, test_size=0.2) #split set
    return train_df, validate_df #return result of split

def build_image_generators(path, batch_size, target_size, train_df, validate_df):
    """
    Build image generators creates ImageDataGenerators for both the training and validation dataframes
    It then uses the flow from dataframe method to set up the training and validation data elegently
    After, the two respective generators are returned.
    params: path -> path to all images, batch_size -> batch size, target_size -> target size of image, train_df -> training dataframe, validate_df -> validation dataframe
    returns: validation generator, training generator
    """
    train_image_generator = ImageDataGenerator( #generates images for training set. will also perform data augmentation
        rescale=1/255.0, #data augmentation
        rotation_range=40, 
        fill_mode='nearest', #helps rotation so there isn't a bunch of black pixels
        shear_range=0.1, 
        horizontal_flip=True,
        brightness_range=[0.4, 1.5])
    validate_image_generator = ImageDataGenerator(rescale=1/255.0) #generates images for validation set. no data augmentation is done here

    train_generator = train_image_generator.flow_from_dataframe( #set up flow from data frame for training
        dataframe=train_df,  #training df
        directory=path,  #path to ALL images
        x_col='filename', 
        y_col='label', 
        batch_size=batch_size, 
        target_size=target_size, 
        color_mode='rgb', #convert all images to RGB 
        shuffle=True, 
        class_mode='binary')

    validation_generator = validate_image_generator.flow_from_dataframe( #set up flow from data frame for validation (similar args as training)
        dataframe=validate_df, #validation dataframe
        directory=path, #path to ALL images
        x_col='filename', 
        y_col='label', 
        batch_size=batch_size, 
        target_size=target_size,
        color_mode='rgb', #all images to RGB
        shuffle=True, 
        class_mode='binary')

    return train_generator, validation_generator #return 

def make_model():
    """
    My neurel network architecture. I had started with a similar number of layers with way less nodes, and played around until this one
    which has given me the best testing and validation accuracy
    params: none
    returns: Sequential keras model
    """
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3))) #layer 1: convolutional layer with 32 nodes and 3x3 kernel
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3, 3), activation='relu')) #layer 2: convolutional layer with 64 nodes and 3x3 kernel
    model.add(MaxPooling2D((2,2)))


    model.add(Conv2D(128, (3, 3), activation='relu')) #layer 3: convolutional layer with 128 nodes and 3x3 kernel
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu')) #layer 4: convolutional layer with 512 nodes and 3x3 kernel
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu')) #layer 5: dense layer 1024 nodes
    
    model.add(BatchNormalization()) #standardize input
    model.add(Dropout(0.5)) #drop out 50% to get rid of some dead nodes
    model.add(Dense(1, activation='sigmoid')) #layer 6: https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/ is why i chose a sigmoid activation function for my output layer and relu for all other layers
    #I was playing around with different numbers of convolutional layers and nodes, this might be a tad overkill but it got me a 92% validation accuracy which is the highest i got when trying a few things so i kept it
    #compile
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy']) #compile model with Adam learning rate and binary cross entropy
    return model #return model

#*********** Driver Function ***********#
def main():
    if len(sys.argv) != 3: #ensure all necessary arguments are guven
        print("Usage: python3 make_nn.py <directory of images> <neurel network name>")
        sys.exit()
    else:
        PATH, NN_NAME = sys.argv[1], sys.argv[2]
        if NN_NAME.endswith(".hd5") == False:
            NN_NAME = NN_NAME + ".hd5" #concatenate .h5 to nn file name so it is saved properly

        train_df, validate_df = process_data(PATH) #create training and validatoin dataframes
        #set up important constants used throughout program
        BATCH_SIZE = 32 
        EPOCHS = 50
        TRAIN_SIZE, VALIDATE_SIZE = train_df.shape[0], validate_df.shape[0] #number of training, validation examples
        STEP_SIZE_TRAIN, STEP_SIZE_VALID = TRAIN_SIZE//BATCH_SIZE, VALIDATE_SIZE//BATCH_SIZE 
        TARGET_SIZE = (100, 100) #size for all images

        train_generator, validation_generator = build_image_generators(PATH, BATCH_SIZE, TARGET_SIZE, train_df, validate_df) #create data generators

        model = make_model() #make model
        # model.summary() #comment back in to see architecture summary 
        history = model.fit( #train model
            train_generator,
            validation_data=validation_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,         
            epochs=EPOCHS,
            validation_steps=STEP_SIZE_VALID       
        )
        model.save(NN_NAME, include_optimizer=False) #save result

if __name__ == "__main__":
    main()
