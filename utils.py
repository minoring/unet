import matplotlib.pyplot as plt


def plot_history(history, epoch):
  loss = history.history['loss']
  accuracy = history.history['accuracy']

  epochs = range(epoch)
  
  plt.figure()
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
