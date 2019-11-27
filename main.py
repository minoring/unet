import os
import imageio

import tensorflow as tf

from absl import app
from absl import flags

from flags import define_flags
from model import unet
from data import train_generator
from utils import load_example
from utils import DisplayCallback
from utils import plot_history
from utils import save_prediction
from utils import create_gif


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

  ## Show model architecture.
  # tf.keras.utils.plot_model(
  #   model,
  #   to_file='model.png',
  #   show_shapes=True
  # )

  example = load_example(flags_obj)
  example_img = imageio.imread('data/membrane/test/0.png')
  # Save first prediction without training.
  save_prediction(model, example_img, example, 0)

  history = model.fit_generator(
      train_gene,
      epochs=flags_obj.epoch,
      steps_per_epoch=flags_obj.steps_per_epoch,
      callbacks=[DisplayCallback(model, example)])

  create_gif()
  plot_history(history, flags_obj.epoch)


def main(_):
  if not os.path.isdir('samples'):
    os.mkdir('samples')
  run(flags.FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)
