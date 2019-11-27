import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate


def unet(flags_obj, n_filters=64):
  # Contracting Path (encoding)
  inputs = Input(flags_obj.input_size)
  conv1 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(n_filters * 2,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(n_filters * 2,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(n_filters * 4,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(n_filters * 4,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(n_filters * 8,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(n_filters * 8,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv4)
  drop4 = Dropout(0.3)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(n_filters * 16,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool4)
  conv5 = Conv2D(n_filters * 16,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv5)
  drop5 = Dropout(0.3)(conv5)

  # Expansive Path (decoding)
  up6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2),
                        padding='same')(drop5)
  merge6 = concatenate([up6, drop4], axis=3)
  conv6 = Conv2D(n_filters * 8,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge6)
  conv6 = Conv2D(n_filters * 8,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv6)

  up7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2),
                        padding='same')(conv6)
  merge7 = concatenate([up7, conv3], axis=3)
  conv7 = Conv2D(n_filters * 4,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge7)
  conv7 = Conv2D(n_filters * 4,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv7)

  up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2),
                        padding='same')(conv7)
  merge8 = concatenate([up8, conv2], axis=3)
  conv8 = Conv2D(n_filters * 2,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge8)
  conv8 = Conv2D(n_filters * 2,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv8)

  up9 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2),
                        padding='same')(conv8)
  merge9 = concatenate([up9, conv1], axis=3)
  conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv9)
  conv9 = Conv2D(2,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv9)

  conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

  model = tf.keras.Model(inputs=inputs, outputs=conv10)

  return model