import os, random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import csv
import keras
import keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, Model
import mltools, dataset2016
import MCLDNN as mcl
import argparse
import tensorflow as tf

if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="MCLDNN")
    parser.add_argument("--epoch", type=int, default=200, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='./RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()
    
    # Set Keras data format as channels_last
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    # Load data
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = dataset2016.load_data(opt.datasetpath, opt.data)
   
 

    # Expand dimensions to match model input
    # Ensure the input shape is [2, 128]
    X_train = np.expand_dims(X_train, axis=3)  # Shape: (batch_size, 2, 128, 1)
    X_val = np.expand_dims(X_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Build and compile model
    model = mcl.MCLDNN(classes=11)

    # Adjust the learning rate
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary to show input and output structure
    model.summary()

    # Define learning rate scheduler
    
    # Train the model
    history = model.fit(X_train, Y_train,
                        batch_size=400,
                        epochs=300,
                        verbose=2,
                        validation_data=(X_val, Y_val),
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(opt.filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_weights_only=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=5, min_lr=0.0000001),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                        ])

    # Evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=opt.batch_size)
    print(score)

    # Ensure the directory exists
    os.makedirs('MCLDNN-master copy/figure', exist_ok=True)

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('MCLDNN-master copy/figure/training_accuracy.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('MCLDNN-master copy/figure/training_loss.png')


