import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, SpatialDropout2D,concatenate, Reshape, Lambda, Flatten, Activation, Attention,GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, GRU, Multiply, Add, Subtract
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau




def cal1(x):
    y = tf.keras.backend.cos(x)
    return y

def cal2(x):
    y = tf.keras.backend.sin(x)
    return y

def MCLDNN(weights=None,
           input_shape=[2, 128],
           classes=11,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    
    input = Input(input_shape + [1], name='input')  # Single input of shape [2, 128, 1]
    
    # Split I/Q channels
    input_I = Lambda(lambda x: x[:, 0, :, :])(input)
    input_Q = Lambda(lambda x: x[:, 1, :, :])(input)
    
    # Flatten and process
    xa = Flatten()(input)
    print("Shape after Flatten:", xa.shape)
    xa = Dense(1, name='fc2',activation='linear')(xa)
    print("Shape after Dense:", xa.shape)
   

    # Simplified calculation using Lambda
    cos1, sin1 = Lambda(lambda x: (tf.keras.backend.cos(x), tf.keras.backend.sin(x)))(xa)
    x11 = Multiply()([input_I, cos1])
    x12 = Multiply()([input_Q, sin1])
    x21 = Multiply()([input_Q, cos1])
    x22 = Multiply()([input_I, sin1])
    y1 = Add()([x11, x12])
    y2 = Subtract()([x21, x22])
    y1 = Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(128, 1), name='reshape2')(y2)
    x11 = concatenate([y1, y2])
    y = Reshape(target_shape=((2, 128, 1)), name='reshape3')(x11)
   
    # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
    x1 = Conv2D(60, (2, 7), padding='valid', activation="relu", name="Conv1-1", kernel_initializer="glorot_uniform")(y)
    print("Shape after Conv1:", x1.shape)
   
    x2 = Conv1D(60, 7, padding='valid', activation="relu", name="Conv1-2", kernel_initializer="glorot_uniform")(input_I)
    x2 = Lambda(lambda x: tf.expand_dims(x, axis=1))(x2)
    print("Shape after Conv2 and expand_dims:", x2.shape)
    
    x3 = Conv1D(60, 7, padding='valid', activation="relu", name="Conv1-3", kernel_initializer="glorot_uniform")(input_Q)
    x3 = Lambda(lambda x: tf.expand_dims(x, axis=1))(x3)
 
  
    # Combine x1, x2, and x3
    x = concatenate([x1,x2,x3], axis=1, name="Concatenate2")
    
  
    print("Shape after Concatenate2:", x.shape)
    
    x = Conv2D(50, (3, 5), padding="valid", activation="relu", name="Conv2", kernel_initializer="glorot_uniform")(x)


    print("Shape after Conv4:", x.shape)
    x = Reshape(target_shape=((118, 50)))(x)
    x = Dropout(0.2)(x)
    x = Conv1D(30, 3, padding="valid", activation="relu", name="Conv3", kernel_initializer="glorot_uniform")(x)
    print("Shape after Conv5:", x.shape)
   
    # Part-B: Temporal Characteristics Extraction Section
    x = Reshape(target_shape=((116, 30)))(x)
    print("Shape after Reshape for GRU:", x.shape)
    x=Dropout(0.2)(x)
    
    x = GRU(units=110, name="GRU", recurrent_activation='sigmoid',use_bias=True)(x)

    print("Shape after GRU:", x.shape)

    x = Dense(classes, activation="softmax", name="Softmax")(x)
    print("Shape after Softmax:", x.shape)

    model = Model(inputs=input, outputs=x)

    # Compile model with learning rate scheduler
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = MCLDNN(classes=11)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
    model.summary()