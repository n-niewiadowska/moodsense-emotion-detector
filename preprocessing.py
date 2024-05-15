import os, random
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory


def preprocessing(DATASET):
  batch_size = 16
  img_height = 48
  img_width = 48

  train_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="training",
    seed=288,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )
  train_data = train_data.map(one_hot_encode)

  validation_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="validation",
    seed=288,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )
  validation_data = validation_data.map(one_hot_encode)

  return train_data, validation_data


def one_hot_encode(image, label):
  label = tf.one_hot(label, depth=7)
  return image, label


## Used when I was making the dataset smaller
# def prepare_data(DATASET):
#   emotions = [emotion for emotion in os.listdir(DATASET)]
#   counts = [len(os.listdir(os.path.join(DATASET, emotion))) for emotion in emotions ]
#   min_count = min(counts)

#   for emotion, count in zip(emotions, counts):
#     if count > min_count:
#       files = os.listdir(os.path.join(DATASET, emotion))
#       random.shuffle(files)
#       for f in files[min_count:]:
#         os.remove(os.path.join(DATASET, emotion, f))
