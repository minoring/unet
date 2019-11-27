from absl import flags


def define_flags():
  flags.DEFINE_string('train_path', 'data/membrane/train',
                      'Path to training data directory')
  flags.DEFINE_string('train_image_dir', 'image',
                      'Directory name of training data input image')
  flags.DEFINE_string('train_label_dir', 'label',
                      'Directory name of training data label')
  flags.DEFINE_list('input_size', [256, 256, 1], 'The size of input image')
  flags.DEFINE_list('target_size', [256, 256], 'Size of target image')
  flags.DEFINE_integer('epoch', 5, 'The number of training epoch')
  flags.DEFINE_integer('batch_size', 3, 'Batch size')
  flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate [3e-4].')
  flags.DEFINE_string(
      'image_color_mode', 'grayscale',
      'Color mode of input image. One of "grayscale", "rgb", "rgba"')
  flags.DEFINE_string('label_color_mode', 'grayscale',
                      'Color mode of label. One of "grayscale", "rgb", "rgba"')
  flags.DEFINE_integer('seed', 1, 'Seed for data augmentation')