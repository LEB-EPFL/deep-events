import numpy as np
import random
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import concatenate, UpSampling2D, BatchNormalization, Reshape
from tensorflow.keras.layers import ConvLSTM2D, Lambda, TimeDistributed #recurrent

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, MeanSquaredError
from tensorflow.keras.regularizers import l2
from deep_events.training_functions import get_loss_function, WMSELoss, WBCELoss
import keras.backend as K
import tensorflow as tf

def get_model_generator(model_name: str = 'simple_recurrent'):
    if model_name == 'simple_recurrent':
        return create_recurrent_model
    elif model_name == 'double_lstm':
        return create_unet_with_two_convlstm
    elif model_name == 'bottleneck_lstm':
        return create_bottleneck_convlstm_model
    elif model_name == 'deep_temporal':
        return create_deep_temporal_unet
    else:
        raise NotImplementedError()

def create_recurrent_model(settings, data_shape, printSummary=False):

    nb_filters = settings.get("nb_filters", 16)
    firstConvSize = settings.get("first_conv_size", 3)
    time_steps = data_shape[-1]

    optimizer_type = Adam(learning_rate=settings.get("initial_learning_rate", 1e-4))
    custom_objects = {settings["loss"]: get_loss_function(settings)}
    loss = list(custom_objects.values())[0]
    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'), WBCELoss(pos_weight=3, name='wbce_loss'), MeanSquaredError()]

    dropout_rate = settings.get("dropout_rate", 0)
    l2_reg = l2(1e-4)

    inputs = Input(shape=(None, None, time_steps))

    def rearrange_fn(x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # (B,T,H,W)
        x = tf.expand_dims(x, axis=-1)          # (B,T,H,W,1)
        return x

    x = Lambda(rearrange_fn)(inputs)

    x = ConvLSTM2D(filters=nb_filters, kernel_size=(3,3), padding='same', return_sequences=False)(x)

    # Encoder
    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg)(x)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Dropout(dropout_rate)(down0)
    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg)(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Dropout(dropout_rate)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(down0)

    down1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Dropout(dropout_rate)(down1)
    down1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Dropout(dropout_rate)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # Center
    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(down1_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)
    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    # Decoder
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(center)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)

    up0 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)
    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    if printSummary:
        model.summary()
    return model


def create_bottleneck_convlstm_model(settings, data_shape, printSummary=False):


    nb_filters = settings.get("nb_filters", 16)
    firstConvSize = settings.get("first_conv_size", 3)
    time_steps = data_shape[-1]

    optimizer_type = Adam(learning_rate=settings.get("initial_learning_rate", 1e-4))
    # loss = settings.get("loss", "binary_crossentropy")
    custom_objects = {settings["loss"]: get_loss_function(settings)}
    loss = list(custom_objects.values())[0]
    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'), WBCELoss(pos_weight=3, name='wbce_loss'), MeanSquaredError()]

    dropout_rate = settings.get("dropout_rate", 0)
    l2_reg = l2(1e-4)

    inputs = Input(shape=(None, None, time_steps))

    def rearrange_fn(x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Add channel dimension: (B, T, H, W, 1)
        x = tf.expand_dims(x, axis=-1)
        return x

    x = Lambda(rearrange_fn)(inputs)

    # down0
    x = TimeDistributed(Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down0 = TimeDistributed(Activation('relu'))(x)
    down0 = TimeDistributed(Dropout(dropout_rate))(down0)
    down0_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(down0)

    # down1
    x = TimeDistributed(Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg))(down0_pool)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down1 = TimeDistributed(Activation('relu'))(x)
    down1 = TimeDistributed(Dropout(dropout_rate))(down1)
    down1_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(down1)
    # shape now: (batch, time, h/4, w/4, nb_filters*2)

    center = ConvLSTM2D(nb_filters*4, (3,3), padding='same', return_sequences=False, kernel_regularizer=l2_reg)(down1_pool)
    # shape now: (batch, h/4, w/4, nb_filters*4)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    def get_last_timestep(x):
        # x is (batch, time, H, W, channels)
        return x[:, -1]  # take the last time step
    down1_last = Lambda(get_last_timestep)(down1)  
    down0_last = Lambda(get_last_timestep)(down0)

    up1 = UpSampling2D((2, 2), interpolation='bilinear')(center)
    up1 = concatenate([down1_last, up1], axis=3)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)

    up0 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    up0 = concatenate([down0_last, up0], axis=3)
    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)
    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)

    if printSummary:
        model.summary()

    return model


def create_unet_with_two_convlstm(settings, data_shape, printSummary=False):

    nb_filters = settings.get("nb_filters", 16)
    first_conv_size = settings.get("first_conv_size", 3)
    time_steps = data_shape[-1]   # number of frames in time
    height = data_shape[1]
    width  = data_shape[2]
    channels = 1  # Adjust if you have multi-channel images

    optimizer_type = Adam(learning_rate=settings.get("initial_learning_rate", 1e-4))
    loss = settings.get("loss", "binary_crossentropy")
    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'), WBCELoss(pos_weight=3, name='wbce_loss'), MeanSquaredError()]

    dropout_rate = settings.get("dropout_rate", 0.0)
    kernel_reg = l2(1e-4)

    # Data initially: (B, H, W, T)
    inputs = Input(shape=(None, None, time_steps))

    def rearrange_fn(x):
        # x: (B, H, W, T) -> (B, T, H, W)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Expand channels dimension -> (B, T, H, W, 1)
        x = tf.expand_dims(x, axis=-1)
        return x

    x = Lambda(rearrange_fn)(inputs)  # (B, T, H, W, 1)

    x = TimeDistributed(Conv2D(nb_filters, (first_conv_size, first_conv_size),
                               padding="same", kernel_regularizer=kernel_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation("relu"))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    x = TimeDistributed(Conv2D(nb_filters, (first_conv_size, first_conv_size),
                               padding="same", kernel_regularizer=kernel_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down0 = TimeDistributed(Activation("relu"))(x)
    down0 = TimeDistributed(Dropout(dropout_rate))(down0)
    down0_pool = TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), padding="same")
    )(down0)

    # down1
    x = TimeDistributed(Conv2D(nb_filters * 2, (3, 3), padding="same",
                               kernel_regularizer=kernel_reg))(down0_pool)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation("relu"))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    x = TimeDistributed(Conv2D(nb_filters * 2, (3, 3), padding="same",
                               kernel_regularizer=kernel_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down1 = TimeDistributed(Activation("relu"))(x)
    down1 = TimeDistributed(Dropout(dropout_rate))(down1)
    down1_pool = TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2), padding="same")
    )(down1)

    # Now shape is (B, T, H/4, W/4, nb_filters*2)

    x = ConvLSTM2D(nb_filters * 4, (3, 3), padding="same",
                   return_sequences=True,  # Keep the time dimension
                   kernel_regularizer=kernel_reg)(down1_pool)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = ConvLSTM2D(nb_filters * 4, (3, 3), padding="same",
                   return_sequences=True,  # Keep it True to maintain time
                   kernel_regularizer=kernel_reg)(x)
    center = BatchNormalization()(x)
    center = Activation("relu")(center)
    center = Dropout(dropout_rate)(center)
    # shape is (B, T, H/4, W/4, nb_filters*4)


    # up1
    up1 = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(center)  # shape: (B, T, H/2, W/2, nb_filters*4)

    up1 = concatenate([down1, up1], axis=-1)  # merges along channels

    up1 = ConvLSTM2D(nb_filters * 2, (3, 3), padding="same",  
                     return_sequences=True,  # still keep time
                     kernel_regularizer=kernel_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation("relu")(up1)
    up1 = Dropout(dropout_rate)(up1)

    # Additional 2D convs
    up1 = TimeDistributed(Conv2D(nb_filters * 2, (3, 3),
                                 padding="same", kernel_regularizer=kernel_reg))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Activation("relu"))(up1)
    up1 = TimeDistributed(Dropout(dropout_rate))(up1)

    # up0
    up0 = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(up1)  # shape: (B, T, H, W, nb_filters*2)

    # Merge with down0: (B, T, H, W, nb_filters)
    up0 = concatenate([down0, up0], axis=-1)

    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), 
                                 padding="same", kernel_regularizer=kernel_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation("relu"))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), 
                                 padding="same", kernel_regularizer=kernel_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation("relu"))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    outputs = TimeDistributed(Conv2D(1, (1, 1), activation="sigmoid"))(up0)
    # outputs shape: (B, T, H, W, 1)

    def last_timestep(x):
        return x[:, -1]  # shape -> (B, H, W, 1)
    outputs = Lambda(last_timestep)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)

    if printSummary:
        model.summary()

    return model

def create_deep_temporal_unet(settings, data_shape, printSummary=False):

    nb_filters = settings.get("nb_filters", 16)
    firstConvSize = settings.get("first_conv_size", 3)
    time_steps = data_shape[-1]

    optimizer_type = Adam(learning_rate=settings.get("initial_learning_rate", 1e-4))

    # Custom loss selection
    custom_objects = {settings["loss"]: get_loss_function(settings)}
    loss = list(custom_objects.values())[0]
    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'),
               WBCELoss(pos_weight=3, name='wbce_loss'),
               tf.keras.metrics.MeanSquaredError()]

    dropout_rate = settings.get("dropout_rate", 0.0)
    l2_reg = l2(1e-4)

    # Input shape: (batch, H, W, T)
    inputs = Input(shape=(None, None, time_steps))

    # Rearrange to (batch, time, height, width, channels=1)
    def rearrange_fn(x):
        # x: (B, H, W, T) -> (B, T, H, W)
        x = tf.transpose(x, perm=[0,3,1,2])
        # Add channel dimension -> (B, T, H, W, 1)
        x = tf.expand_dims(x, axis=-1)
        return x

    x = Lambda(rearrange_fn)(inputs)  # shape: (B, T, H, W, 1)


    x = TimeDistributed(Conv2D(nb_filters, (firstConvSize, firstConvSize), 
                               padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    x = TimeDistributed(Conv2D(nb_filters, (firstConvSize, firstConvSize), 
                               padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down0 = TimeDistributed(Activation('relu'))(x)
    down0 = TimeDistributed(Dropout(dropout_rate))(down0)

    down0_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2), padding='same'))(down0)

    # down1
    x = TimeDistributed(Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg))(down0_pool)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    x = TimeDistributed(Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg))(x)
    x = TimeDistributed(BatchNormalization())(x)
    down1 = TimeDistributed(Activation('relu'))(x)
    down1 = TimeDistributed(Dropout(dropout_rate))(down1)

    down1_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2), padding='same'))(down1)
    # shape => (B, T, H/4, W/4, nb_filters*2)
---
    x = ConvLSTM2D(nb_filters*4, (3,3), padding='same', 
                   return_sequences=True, kernel_regularizer=l2_reg)(down1_pool)
    x = ConvLSTM2D(nb_filters*4, (3,3), padding='same', 
                   return_sequences=True, kernel_regularizer=l2_reg)(x)
    # Now we have (B, T, H/4, W/4, nb_filters*4)

    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    # We can also optionally do a TimeDistributed(Conv2D) here

    # up1
    up1 = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)  # shape => (B, T, H/2, W/2, nb_filters*4)
    up1 = concatenate([down1, up1], axis=-1)  # merges along channels

    up1 = ConvLSTM2D(nb_filters*2, (3,3), padding='same', return_sequences=True, kernel_regularizer=l2_reg)(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Activation('relu'))(up1)
    up1 = TimeDistributed(Dropout(dropout_rate))(up1)

    up1 = TimeDistributed(Conv2D(nb_filters*2, (3,3), padding='same', kernel_regularizer=l2_reg))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Activation('relu'))(up1)
    up1 = TimeDistributed(Dropout(dropout_rate))(up1)

    # up0
    up0 = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(up1)  # shape => (B, T, H, W, nb_filters*2)
    up0 = concatenate([down0, up0], axis=-1)

    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation('relu'))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation('relu'))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    outputs = TimeDistributed(Conv2D(1, (1,1), activation='sigmoid'))(up0)

    outputs = Lambda(lambda z: z[:, -1], name='last_frame')(outputs)
    # That yields (B, H, W, 1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)

    if printSummary:
        model.summary()

    return model


if __name__ == '__main__':
    data_shape = (16, 128, 128, 5)
    settings = {
        "nb_filters": 16,
        "first_conv_size": 12,
        "initial_learning_rate": 1e-4,
        "loss": "binary_crossentropy"
    }
    model = create_deep_temporal_unet(settings, data_shape, printSummary=True)