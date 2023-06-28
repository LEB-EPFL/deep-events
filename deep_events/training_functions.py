import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.layers import concatenate, UpSampling2D, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, MeanSquaredError
import keras.backend as K
import tensorflow as tf

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
def soft_dice_loss(y_true, y_pred, smooth=1e-7):
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




def create_model(settings, data_shape, printSummary=False, ):
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    #Hyperparameters
    optimizer_type = Adam(learning_rate=0.5e-3)
    custom_objects = {settings["loss"]: get_loss_function(settings)}
    loss = list(custom_objects.values())[0] #dice_loss # 'binary_crossentropy'  # 'mse'

    metrics = [BinaryAccuracy(), MeanSquaredError()]

    #Network architecture
    if len(data_shape) > 3:
        settings["nb_input_channels"] = data_shape[-1]
    input_shape = (None, None, settings["nb_input_channels"])
    print(f"INPUT SHAPE {input_shape}")
    inputs = Input(shape=input_shape)

    if settings["loss"] == "soft_dice":
        final_activation = "linear"
    else:
        final_activation = "sigmoid"

    # Encoder
    print('* Start Encoder Section *')

    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(down0)

    down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # Center
    print('* Start Center Section *')
    center = Conv2D(nb_filters*4, (3, 3), padding='same')(down1_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(nb_filters*4, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)

    up1 = UpSampling2D((2, 2))(center)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)

    outputs = Conv2D(1, (1, 1), activation=final_activation)(up0)  # was relu also before
    outputs.set_shape([None, None, None, 1])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    if printSummary:
        print(model.summary())
    return model


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
    elif loss == "mse":
        return "mse"
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
