from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def preprocess_data(img, mask):
  img = img / 255.
  mask = mask / 255.
  mask[mask > 0.5] = 1
  mask[mask <= 0.5] = 0

  return (img, mask)


def train_generator(flags_obj, data_aug_args):
  """Create training example generator flow from directory
  Args:
    flags_obj: absl flag object.
    data_aug_args: Arguments for tf.keras.preprocessing.image ImageDataGenerator

  Yields:
    tf.data.Dataset of (img, mask)
  """

  image_datagen = ImageDataGenerator(**data_aug_args)
  mask_datagen = ImageDataGenerator(**data_aug_args)
  image_generator = image_datagen.flow_from_directory(
      directory=flags_obj.train_path,
      classes=[flags_obj.train_image_dir],
      class_mode=None,
      color_mode=flags_obj.image_color_mode,
      target_size=flags_obj.target_size,
      batch_size=flags_obj.batch_size,
      save_to_dir=None,
      save_prefix='image',
      seed=flags_obj.seed  # Same seed for image_datagen and mask_datagen.
  )
  mask_generator = mask_datagen.flow_from_directory(
      directory=flags_obj.train_path,
      classes=[flags_obj.train_label_dir],
      class_mode=None,
      color_mode=flags_obj.label_color_mode,
      target_size=flags_obj.target_size,
      batch_size=flags_obj.batch_size,
      save_to_dir=None,
      save_prefix='label',
      seed=flags_obj.seed)

  train_gene = (preprocess_data(img, mask)
                for img, mask in zip(image_generator, mask_generator))

  return train_gene
