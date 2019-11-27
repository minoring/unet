import tensorflow as tf

from absl import app
from absl import flags

from flags import define_flags
from model import unet
from data import train_generator


def run(flags_obj):
  data_aug_args = dict(rotation_range=0.2,
                       width_shift_range=0.05,
                       height_shift_range=0.05,
                       shear_range=0.05,
                       zoom_range=0.05,
                       horizontal_flip=True,
                       fill_mode='nearest')

  train_gene = train_generator(flags_obj, data_aug_args)

  model = unet(flags_obj, n_filters=64)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=flags_obj.learning_rate),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy'])

  H = model.fit_generator(train_gene,
                          epochs=flags_obj.epoch,
                          steps_per_epoch=300)
  

def main(_):
  run(flags.FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)