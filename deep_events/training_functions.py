import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import concatenate, UpSampling2D, BatchNormalization, Reshape, UpSampling3D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, MeanSquaredError
from tensorflow.keras.regularizers import l2
import keras.backend as K
import tensorflow as tf



tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1.):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1. - dice_coef

    return dice_loss

@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):

    # calculate focal loss coefficients
    ones = tf.ones_like(y_true)
    p_t = tf.where(tf.equal(y_true, ones), y_pred, ones-y_pred)
    alpha_factor = tf.where(tf.equal(y_true, ones), alpha, ones-alpha)
    modulating_factor = tf.pow(1.0-p_t, gamma)
    focal_loss = -alpha_factor * modulating_factor * tf.math.log(tf.clip_by_value(p_t, K.epsilon(), 1.0))

    return tf.reduce_mean(focal_loss, axis=-1)

@tf.keras.utils.register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + smooth) / (denominator + smooth)


@tf.keras.utils.register_keras_serializable()
def soft_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.ones_like(y_true) * alpha
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = alpha_factor * tf.pow(1 - pt, gamma)
    cross_entropy = -tf.math.log(pt)
    loss = focal_weight * cross_entropy
    return tf.reduce_mean(loss, axis=[1,2])

@tf.keras.utils.register_keras_serializable()
class WBCELoss(tf.keras.losses.Loss):
    def __init__(self, pos_weight=1, name='wbce_loss'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    def call(self, y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        # Apply the weights
        weight_vector = y_true * self.pos_weight + (1. - y_true)
        weighted_bce = weight_vector * bce
        return tf.keras.backend.mean(weighted_bce)

    def get_config(self):
        return {'pos_weight': self.pos_weight}    
tf.keras.utils.get_custom_objects().update({'WBCELoss': WBCELoss})
 

@tf.keras.utils.register_keras_serializable(package='deep_events', name='wmse_loss')
class WMSELoss(tf.keras.losses.Loss):
    def __init__(self, pos_weight=1, name='wmse_loss'):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        weight_vector = y_true * self.pos_weight + (1. - y_true)
        return tf.reduce_mean(weight_vector * tf.square(y_true - y_pred))

    def get_config(self):
        return {'pos_weight': self.pos_weight}
tf.keras.utils.get_custom_objects().update({'WMSELoss': WMSELoss})

def create_model(settings, data_shape, printSummary=False):
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    #Hyperparameters
    optimizer_type = Adam(learning_rate=0.5e-3)

    if settings["loss"] == "wmse":
        loss = WMSELoss(pos_weight=settings.get("weight", 10))
    else:
        custom_objects = {settings["loss"]: get_loss_function(settings)}
        loss = list(custom_objects.values())[0] #dice_loss # 'binary_crossentropy'  # 'mse'

    metrics = [WMSELoss(pos_weight=3, name='wmse'), WBCELoss(pos_weight=3, name='wbce'), MeanSquaredError()]

    if settings["loss"] == "soft_dice":
        final_activation = "sigmoid" #"linear"
    else:
        final_activation = "sigmoid"


    dropout_rate = settings.get("dropout_rate", 0)
    l2_reg = l2(1e-4)

    # Assuming data_shape is (batch_size, height, width, timepoints)
    input_shape = (*data_shape, 1)  # Add channel dimension
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(nb_filters*2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(p1)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv3D(nb_filters*2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(nb_filters*4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(p2)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv3D(nb_filters*4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    # Bottleneck
    c4 = Conv3D(nb_filters*8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(p3)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv3D(nb_filters*8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c4)

    # Decoder
    u5 = UpSampling3D((2, 2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv3D(nb_filters*4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(u5)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv3D(nb_filters*4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c5)

    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv3D(nb_filters*2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv3D(nb_filters*2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c6)

    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(c7)

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)
    outputs = Reshape((data_shape[1], data_shape[2], data_shape[3]))(outputs)  # Reshape to match original 3D output shape



    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    if printSummary:
        print(model.summary())
    return model

def conv_block(x, filters, kernel_size, l2_reg, dropout_rate, dim):
    if dim == 3:
        x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2_reg)(x)
    else:
        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    return x

def pool_block(x, pool_size, dim):
    if dim == 3:
        x = MaxPooling3D(pool_size, strides=pool_size, padding='same')(x)
    else:
        x = MaxPooling2D(pool_size, strides=pool_size, padding='same')(x)
    return x

def upsample_block(x, size, dim):
    if dim == 3:
        x = UpSampling3D(size)(x)
    else:
        x = UpSampling2D(size)(x)
    return x

def get_loss_function(settings: dict = None):
    if settings is None:
        return "binary_crossentropy"

    loss = settings["loss"]
    if loss == "dice":
        return dice_loss
    elif loss == "focal":
        return focal_loss
    elif loss == "soft_dice":
        return soft_dice_loss
    elif loss == "soft_focal":
        return soft_focal_loss
    elif loss == "wbce":
        return WBCELoss(settings["weight"])
    elif loss == "mse":
        return "mse"
    elif loss == "wmse":
        return WMSELoss(settings["weight"])
    elif loss == "binary_crossentropy":
        return "binary_crossentropy"
    else:
        NotImplementedError(f"{loss} has not been implemented as loss function in this framework.")




def train_model(model, input_data, output_data, batch_size, validtrain_split_ratio):
    # Split dataset into [test] and [train+valid]
    max_epochs = 20  # maxmimum number of epochs to be iterated
    batch_shuffle= True   # shuffle the training data prior to batching before each epoch

    history = model.fit(input_data, output_data,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        validation_split=validtrain_split_ratio,
                        shuffle=batch_shuffle,
                        verbose=2)

    return history.history

