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
    """
    Create a U-Net-like model with a ConvLSTM2D layer to handle temporal sequences of images.
    The input data is originally (batch, height, width, time), and we rearrange it within the model
    to (batch, time, height, width, channels) before passing through the ConvLSTM2D layer.
    """

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

    # Original input: (batch, height, width, time)
    inputs = Input(shape=(None, None, time_steps))

    # Rearrange to (batch, time, height, width, channels)
    # Using a Lambda layer with tf operations:
    # (B,H,W,T) -> transpose to (B,T,H,W)
    # then expand dims to add channels: (B,T,H,W,1)
    def rearrange_fn(x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # (B,T,H,W)
        x = tf.expand_dims(x, axis=-1)          # (B,T,H,W,1)
        return x

    x = Lambda(rearrange_fn)(inputs)

    # Temporal modeling with ConvLSTM2D
    x = ConvLSTM2D(filters=nb_filters, kernel_size=(3,3), padding='same', return_sequences=False)(x)
    # Now x is (batch, height, width, nb_filters)

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
    """
    U-Net with ConvLSTM2D at the bottleneck.
    
    Assumptions:
    - Input data shape: (batch, height, width, time)
    - Single channel images (channels=1)
    - We process each frame through the encoder using TimeDistributed layers.
    - At the bottleneck, we apply ConvLSTM2D across the time dimension.
    - The output is a single segmentation mask (for the last time step or aggregated info).
    """

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

    # (batch, height, width, time)
    inputs = Input(shape=(None, None, time_steps))

    # (batch, time, height, width, channel)
    def rearrange_fn(x):
        # x: (B, H, W, T) -> (B, T, H, W)
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

    # --- Bottleneck with ConvLSTM2D ---
    # ConvLSTM2D across the time dimension to integrate temporal info.
    # ConvLSTM2D expects (batch, time, height, width, channels)
    center = ConvLSTM2D(nb_filters*4, (3,3), padding='same', return_sequences=False, kernel_regularizer=l2_reg)(down1_pool)
    # shape now: (batch, h/4, w/4, nb_filters*4)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    # --- Decoder (no time dimension now, we're dealing with a single frame) ---
    # last "down" features from the encoder. Choose frame from  TimeDistributed
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
    """
    U-Net-like architecture that:
      1) Uses TimeDistributed conv/pool in the encoder,
      2) Applies a ConvLSTM2D in the bottleneck (return_sequences=True),
      3) Applies another ConvLSTM2D in the decoder,
      4) Keeps the time dimension throughout,
      5) Produces a per-frame (time) output or a single-frame output (configurable).

    Args:
        settings (dict): Hyperparameters (nb_filters, first_conv_size, dropout_rate, etc.)
        data_shape (tuple): (batch, height, width, time)
        print_summary (bool): Whether to print the model summary.
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """

    # --- Extract parameters ---
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

    # --- 1) Define Model Input ---
    # Data initially: (B, H, W, T)
    inputs = Input(shape=(None, None, time_steps))

    # Rearrange to (B, T, H, W, C)
    def rearrange_fn(x):
        # x: (B, H, W, T) -> (B, T, H, W)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Expand channels dimension -> (B, T, H, W, 1)
        x = tf.expand_dims(x, axis=-1)
        return x

    x = Lambda(rearrange_fn)(inputs)  # (B, T, H, W, 1)

    # --- 2) ENCODER with TimeDistributed ---
    # down0
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

    # --- 3) BOTTLENECK with ConvLSTM2D (return_sequences=True) ---
    # This keeps the time dimension for subsequent layers in the decoder.
    x = ConvLSTM2D(nb_filters * 4, (3, 3), padding="same",
                   return_sequences=True,  # Keep the time dimension
                   kernel_regularizer=kernel_reg)(down1_pool)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    # Optional second ConvLSTM2D or standard Conv2D in the bottleneck
    x = ConvLSTM2D(nb_filters * 4, (3, 3), padding="same",
                   return_sequences=True,  # Keep it True to maintain time
                   kernel_regularizer=kernel_reg)(x)
    center = BatchNormalization()(x)
    center = Activation("relu")(center)
    center = Dropout(dropout_rate)(center)
    # shape is (B, T, H/4, W/4, nb_filters*4)

    # --- 4) DECODER ---
    # We'll do time-distributed upsampling & merges with the corresponding skip connections.
    # Then we'll insert another ConvLSTM2D to refine spatiotemporal info at the decoded resolution.

    # up1
    up1 = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(center)  # shape: (B, T, H/2, W/2, nb_filters*4)

    # Concatenate skip from down1 (which has shape (B, T, H/2, W/2, nb_filters*2))
    # Must match channels dimension, so axis=-1 (the channels axis in (B,T,H,W,C))
    up1 = concatenate([down1, up1], axis=-1)  # merges along channels

    # Now apply a ConvLSTM2D in the decoder:
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

    # (Optional) Another ConvLSTM2D here if you'd like even more temporal modeling
    # For simplicity, let's do standard 2D conv
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

    # --- 5) OUTPUT LAYER ---
    # Option A: produce mask per time step => shape: (B, T, H, W, 1)
    # Option B: produce mask for last time step only => shape: (B, H, W, 1)

    # For demonstration, let's produce a mask for EVERY time step:
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation="sigmoid"))(up0)
    # outputs shape: (B, T, H, W, 1)

    # If you only want the mask for the final time step, do something like:
    def last_timestep(x):
        return x[:, -1]  # shape -> (B, H, W, 1)
    outputs = Lambda(last_timestep)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)

    if printSummary:
        model.summary()

    return model

def create_deep_temporal_unet(settings, data_shape, printSummary=False):
    """
    A U-Net-like model that:
      - Processes time dimension with TimeDistributed in the encoder
      - Uses two ConvLSTM2D layers in the bottleneck (both return_sequences=True)
      - Preserves time dimension in the decoder with TimeDistributed(UpSampling2D)
      - Optionally uses a ConvLSTM2D in the decoder for spatiotemporal refinement
      - Finally outputs either a per-frame (B, T, H, W, 1) segmentation or a single (B, H, W, 1)

    Assumptions:
      - data_shape = (batch, height, width, time)
      - single-channel frames
    """

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

    # ----------------
    # Encoder
    # ----------------
    # down0
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

    # ----------------
    # Bottleneck with TWO ConvLSTM2D layers (both return_sequences=True)
    # so we exit with shape (B, T, H/4, W/4, nb_filters*4)
    # ----------------
    x = ConvLSTM2D(nb_filters*4, (3,3), padding='same', 
                   return_sequences=True, kernel_regularizer=l2_reg)(down1_pool)
    x = ConvLSTM2D(nb_filters*4, (3,3), padding='same', 
                   return_sequences=True, kernel_regularizer=l2_reg)(x)
    # Now we have (B, T, H/4, W/4, nb_filters*4)

    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    # We can also optionally do a TimeDistributed(Conv2D) here

    # ----------------
    # Decoder
    # We'll do time-distributed upsampling
    # and we can also incorporate a ConvLSTM2D for spatiotemporal refinement
    # ----------------

    # up1
    up1 = TimeDistributed(UpSampling2D((2,2), interpolation='bilinear'))(x)  # shape => (B, T, H/2, W/2, nb_filters*4)
    # Concatenate skip (down1) which is shape (B, T, H/2, W/2, nb_filters*2)
    up1 = concatenate([down1, up1], axis=-1)  # merges along channels

    # Optional spatiotemporal refinement with ConvLSTM2D
    up1 = ConvLSTM2D(nb_filters*2, (3,3), padding='same', return_sequences=True, kernel_regularizer=l2_reg)(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Activation('relu'))(up1)
    up1 = TimeDistributed(Dropout(dropout_rate))(up1)

    # Additional TimeDistributed(Conv2D)
    up1 = TimeDistributed(Conv2D(nb_filters*2, (3,3), padding='same', kernel_regularizer=l2_reg))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Activation('relu'))(up1)
    up1 = TimeDistributed(Dropout(dropout_rate))(up1)

    # up0
    up0 = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(up1)  # shape => (B, T, H, W, nb_filters*2)
    # Merge with down0 (B, T, H, W, nb_filters)
    up0 = concatenate([down0, up0], axis=-1)

    # We'll do standard conv now (no LSTM)
    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation('relu'))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    up0 = TimeDistributed(Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg))(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(Activation('relu'))(up0)
    up0 = TimeDistributed(Dropout(dropout_rate))(up0)

    # -------------
    # Final Output
    # -------------
    # Option A: produce a segmentation mask for EVERY time step => shape (B, T, H, W, 1)
    outputs = TimeDistributed(Conv2D(1, (1,1), activation='sigmoid'))(up0)

    # Option B: if you only want the mask at the last time step, do:
    outputs = Lambda(lambda z: z[:, -1], name='last_frame')(outputs)
    # That yields (B, H, W, 1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)

    if printSummary:
        model.summary()

    return model


if __name__ == '__main__':
    # Example usage:
    # Original data shape: (batch, height, width, time)
    data_shape = (16, 128, 128, 5)
    settings = {
        "nb_filters": 16,
        "first_conv_size": 12,
        "initial_learning_rate": 1e-4,
        "loss": "binary_crossentropy"
    }
    model = create_deep_temporal_unet(settings, data_shape, printSummary=True)