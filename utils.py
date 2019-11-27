import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
import glob


class DisplayCallback(tf.keras.callbacks.Callback):

  def __init__(self, model, example):

    self.model = model
    self.example = example

  def on_epoch_end(self, epoch, logs=None):
    save_prediction(self.model, self.example, epoch + 1)


def save_prediction(model, example, epoch):
  image, mask = example
  pred_mask = model.predict(tf.expand_dims(image, axis=0))

  plt.figure(figsize=(15, 15))

  plt.subplot(1, 3, 1)
  plt.title('Input image')
  plt.imshow(tf.keras.preprocessing.image.array_to_img(image), cmap='gray')

  plt.subplot(1, 3, 2)
  plt.title('True mask')
  plt.imshow(tf.keras.preprocessing.image.array_to_img(mask), cmap='gray')

  plt.subplot(1, 3, 3)
  plt.title('Predicted mask')
  plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]),
             cmap='gray')

  plt.savefig('samples/predict_image_{}.jpg'.format(epoch))
  plt.clf()


def plot_history(history, epoch):
  loss = history.history['loss']
  accuracy = history.history['accuracy']

  epochs = range(epoch)

  plt.figure(figsize=(15, 15))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.xlabel('Epoch')
  plt.ylabel('Binary crossentropy loss')
  plt.ylim([0, 1])

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, 'b', label='Training accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])

  plt.savefig('Learning history')
  plt.show()


def load_example_img(flags_obj):
  example_img = tf.io.read_file('data/membrane/test/0.png')
  example_img = tf.image.decode_png(example_img, channels=1)
  example_img = tf.image.convert_image_dtype(example_img, tf.float32)
  example_img = tf.image.resize(
      example_img, (flags_obj.input_size[0], flags_obj.input_size[1]))
  example_mask = tf.io.read_file('data/membrane/test/0_predict.png')
  example_mask = tf.image.decode_png(example_mask, channels=1)
  example_mask = tf.image.convert_image_dtype(example_mask, tf.float32)
  example_img = tf.image.resize(
      example_mask, (flags_obj.input_size[0], flags_obj.input_size[1]))

  return (example_img, example_mask)


def create_gif():
  """Create gif using saved images"""
  anim_file = 'samples/unet.gif'

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('samples/predict_image_*.jpg')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i**0.5)
      if round(frame) > round(last):
        last = frame
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)