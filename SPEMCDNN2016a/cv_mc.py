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

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os

class ComplexConv2D(tf.keras.layers.Layer):
    """Custom complex-valued convolution layer"""
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        # 输入形状：[batch, 2, seq_len, 1]
        input_dim = input_shape[-1]
        
        # 实部和虚部的卷积核
        self.kernel_real = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.filters),
            initializer='glorot_uniform',
            name='kernel_real'
        )
        self.kernel_imag = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.filters),
            initializer='glorot_uniform',
            name='kernel_imag'
        )
        
        # 偏置项
        self.bias_real = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            name='bias_real'
        )
        self.bias_imag = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            name='bias_imag'
        )

    def call(self, inputs):
        # 分离IQ通道
        input_real = inputs[:, 0:1, :, :]  # I通道 [batch, 1, seq, 1]
        input_imag = inputs[:, 1:2, :, :]  # Q通道 [batch, 1, seq, 1]
        
        # 执行复数卷积
        conv_real = tf.nn.conv2d(input_real, self.kernel_real, strides=[1, 1, 1, 1], padding='VALID')
        conv_imag = tf.nn.conv2d(input_imag, self.kernel_imag, strides=[1, 1, 1, 1], padding='VALID')

        # 复数卷积计算
        output_real = conv_real - conv_imag
        output_imag = conv_real + conv_imag
        
        # 合并结果并添加偏置
        output = tf.concat([
            output_real + self.bias_real,
            output_imag + self.bias_imag
        ], axis=1)
        
        if self.activation:
            output = tf.keras.activations.get(self.activation)(output)
        return output

    def compute_output_shape(self, input_shape):
        # 计算输出形状
        return (input_shape[0], input_shape[1] - self.kernel_size[0] + 1, input_shape[2] - self.kernel_size[1] + 1, self.filters)

def CV_MC(weights=None,
           input_shape=[2, 128],
           classes=11,
           **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    # 输入层
    input = Input(shape=input_shape + [1], name='input')  # [2, 128, 1]
    
    # 使用复数卷积替代相位旋转模块
    x = ComplexConv2D(filters=1, 
                    kernel_size=(2, 1),  # 处理IQ两个通道
                    activation='linear')(input)  # 输出保持[2, 128, 1]

    # Part-A: 多通道特征提取
    # 分支1: 复数卷积处理后的IQ数据
    x1 = Conv2D(60, (2, 7), padding='valid', activation="relu",
               name="Conv1-1", kernel_initializer="glorot_uniform")(x)
    
    # 分支2: I通道处理
    input_I = Lambda(lambda x: x[:, 0:1, :, :])(input)
    x2 = Conv1D(60, 7, padding='valid', activation="relu",
               name="Conv1-2", kernel_initializer="glorot_uniform")(input_I[:,0,:,:])
    x2 = Reshape((1, 122, 60))(x2)  # 保持维度对齐
    
    # 分支3: Q通道处理
    input_Q = Lambda(lambda x: x[:, 1:2, :, :])(input)
    x3 = Conv1D(60, 7, padding='valid', activation="relu",
               name="Conv1-3", kernel_initializer="glorot_uniform")(input_Q[:,0,:,:])
    x3 = Reshape((1, 122, 60))(x3)
    
    # 合并多分支特征
    x = concatenate([x1, x2, x3], axis=1)
    
    # 后续特征处理
    x = Conv2D(50, (3, 5), padding="valid", activation="relu",
              name="Conv2", kernel_initializer="glorot_uniform")(x)
    x = Reshape((118, 50))(x)
    x = Dropout(0.2)(x)
    x = Conv1D(30, 3, padding="valid", activation="relu",
              name="Conv3", kernel_initializer="glorot_uniform")(x)
    
    # Part-B: 时序特征提取
    x = GRU(110, recurrent_activation='sigmoid', use_bias=True)(x)
    x = Dense(classes, activation="softmax")(x)
    
    model = Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(0.001),
                 metrics=['accuracy'])
    
    if weights is not None:
        model.load_weights(weights)
        
    return model

if __name__ == '__main__':
    model = CV_MC(classes=11)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
    model.summary()