import numpy as np
import random
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import concatenate, UpSampling2D, BatchNormalization, Reshape
from tensorflow.keras.layers import ConvLSTM2D, Lambda, TimeDistributed #recurrent

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, MeanSquaredError
from tensorflow.keras.regularizers import l2
import keras.backend as K
import tensorflow as tf


# set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
class SoftFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='soft_focal'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)
        focal_weight = tf.pow((1.0 - pt), self.gamma)
        ce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(alpha_factor * focal_weight * ce)

    def get_config(self):
        c = super().get_config()
        c.update({'alpha': self.alpha, 'gamma': self.gamma})
        return c
tf.keras.utils.get_custom_objects().update({'soft_focal': SoftFocalLoss})

@tf.keras.utils.register_keras_serializable()
class WBCELoss(tf.keras.losses.Loss):
    def __init__(self, pos_weight=1, name='wbce_loss'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
        print("WBCE WEIGHT", pos_weight)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        # Apply the weights
        weight_vector = y_true * self.pos_weight + (1. - y_true)

        weighted_bce = weight_vector * bce
        return tf.reduce_mean(weighted_bce)

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
    optimizer_type = Adam(learning_rate=settings["initial_learning_rate"])

    if settings["loss"] == "wmse":
        loss = WMSELoss(pos_weight=settings.get("weight", 10))
    else:
        custom_objects = {settings["loss"]: get_loss_function(settings)}
        loss = list(custom_objects.values())[0] #dice_loss # 'binary_crossentropy'  # 'mse'

    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'), WBCELoss(pos_weight=3, name='wbce_loss'), MeanSquaredError()]

    #Network architecture
    if len(data_shape) > 3:
        settings["nb_input_channels"] = data_shape[-1]
    input_shape = (None, None, settings["nb_input_channels"])
    print(f"INPUT SHAPE {input_shape}")
    inputs = Input(shape=input_shape)

    if settings["loss"] == "soft_dice":
        final_activation = "sigmoid" #"linear"
    else:
        final_activation = "sigmoid"
    dropout_rate = settings.get("dropout_rate", 0)
    l2_reg = l2(1e-4)
    #kernel regularizer added 20240731
    
    # Encoder
    print(f'* Start Encoder Section dropout: {dropout_rate}*')

    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg)(inputs)
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
    print('* Start Center Section *')
    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(down1_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)
    center = Conv2D(nb_filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    up1 = UpSampling2D((2, 2), interpolation='bilinear')(center)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1= Dropout(dropout_rate)(up1)

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

    outputs = Conv2D(1, (1, 1), activation=final_activation)(up0)  # was relu also before
    outputs.set_shape([None, None, None, 1])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    if printSummary:
        print(model.summary())
    return model


class ChannelSlice(Layer):
    def __init__(self, idx, **kwargs):
        super(ChannelSlice, self).__init__(**kwargs)
        self.idx = idx

    def call(self, inputs):
        # Slices out the channel at index self.idx and keeps the dimension.
        return inputs[..., self.idx:self.idx+1]

    def get_config(self):
        config = super(ChannelSlice, self).get_config()
        config.update({'idx': self.idx})
        return config

def create_distributed_model(settings, data_shape, printSummary=False):
    nb_filters, firstConvSize = settings["nb_filters"], settings["first_conv_size"]
    #Hyperparameters
    optimizer_type = Adam(learning_rate=settings["initial_learning_rate"])

    if settings["loss"] == "wmse":
        loss = WMSELoss(pos_weight=settings.get("weight", 10))
    else:
        custom_objects = {settings["loss"]: get_loss_function(settings)}
        loss = list(custom_objects.values())[0] #dice_loss # 'binary_crossentropy'  # 'mse'

    metrics = [WMSELoss(pos_weight=3, name='wmse_loss'), WBCELoss(pos_weight=3, name='wbce_loss'), MeanSquaredError()]

    #Network architecture
    if len(data_shape) > 3:
        settings["nb_input_channels"] = data_shape[-1]
    input_shape = (None, None, settings["nb_input_channels"])
    print(f"INPUT SHAPE {input_shape}")
    inputs = Input(shape=input_shape)

    if settings["loss"] == "soft_dice":
        final_activation = "sigmoid" #"linear"
    else:
        final_activation = "sigmoid"
    dropout_rate = settings.get("dropout_rate", 0)
    l2_reg = l2(1e-4)
        # We will store outputs for skip connections from each branch.
    down0_list = []       # from first block (before pooling)
    down1_list = []       # from second block (before pooling)
    down1_pool_list = []  # pooling outputs to feed into the center

    # Note: iterate over the number of channels.
    for i in range(settings["nb_input_channels"]):
        # Extract the i-th channel as a (None, None, 1) tensor
        channel = ChannelSlice(idx=i, name=f'channel_slice_{i}')(inputs)
        
        # --- Down0 block for channel i ---
        # Two successive convolutions with dropout and BN.
        x = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg)(channel)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same', kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Save the output for skip connection at level 0 for this channel.
        down0_list.append(x)
        
        # Pooling for the next block.
        x_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
        
        # --- Down1 block for channel i ---
        x = Conv2D(nb_filters * 2, (3, 3), padding='same', kernel_regularizer=l2_reg)(x_pool)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv2D(nb_filters * 2, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Save the pre-pooling output for the skip connection at level 1.
        down1_list.append(x)
        
        # Pool again to go deeper.
        x_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
        down1_pool_list.append(x_pool)

    # Merge (concatenate) the outputs from all channels.
    # The channels remain separate in their own branches until here.
    merged_down0 = concatenate(down0_list, axis=-1)
    merged_down1 = concatenate(down1_list, axis=-1)
    merged_down1_pool = concatenate(down1_pool_list, axis=-1)

    # --- Center (Bottleneck) ---
    print('* Start Center Section *')
    center = Conv2D(nb_filters * 4, (3, 3), padding='same', kernel_regularizer=l2_reg)(merged_down1_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    center = Conv2D(nb_filters * 4, (3, 3), padding='same', kernel_regularizer=l2_reg)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Dropout(dropout_rate)(center)

    # --- Decoder ---
    # First upsampling + skip connection from level 1
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(center)
    # Use the merged down1 skip connection.
    up1 = concatenate([merged_down1, up1], axis=-1)
    up1 = Conv2D(nb_filters * 2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)

    up1 = Conv2D(nb_filters * 2, (3, 3), padding='same', kernel_regularizer=l2_reg)(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(dropout_rate)(up1)

    # Second upsampling + skip connection from level 0
    up0 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    # Use the merged down0 skip connection.
    up0 = concatenate([merged_down0, up0], axis=-1)
    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)

    up0 = Conv2D(nb_filters, (3, 3), padding='same', kernel_regularizer=l2_reg)(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Dropout(dropout_rate)(up0)

    # --- Output ---
    outputs = Conv2D(1, (1, 1), activation=final_activation)(up0)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
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
        print('Using soft focal loss')
        return SoftFocalLoss(settings['weight']['alpha'], settings['weight']['gamma'])
    elif loss == "wbce_loss":
        print('Using wbce loss')
        return WBCELoss(settings["weight"])
    elif loss == "mse":
        return "mse"
    elif loss == "wmse_loss":
        return WMSELoss(settings["weight"])
    elif loss == "binary_crossentropy":
        return "binary_crossentropy"
    else:
        NotImplementedError(f"{loss} has not been implemented as loss function in this framework.")

def get_model_generator(model_name: str = 'simple_recurrent'):
    from deep_events import lstm_models
    if model_name == 'unet':
        return create_model
    if model_name == 'unet_distributed':
        return create_distributed_model
    elif model_name == 'simple_recurrent':
        return lstm_models.create_recurrent_model
    elif model_name == 'double_lstm':
        return lstm_models.create_unet_with_two_convlstm
    elif model_name == 'bottleneck_lstm':
        return lstm_models.create_bottleneck_convlstm_model
    elif model_name == 'deep_temporal':
        return lstm_models.create_deep_temporal_unet
    else:
        raise NotImplementedError()

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
    model = create_model(settings, data_shape, printSummary=True)
