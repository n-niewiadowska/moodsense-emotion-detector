import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def make_plots(model, history, val_data):
  os.makedirs("plots", exist_ok=True)
  cm, accuracy = calculate_confusion_matrix_and_accuracy(model, val_data)

  visualize_confusion_matrix(cm)
  print(f"||   Accuracy of the model: {accuracy * 100:.2f}%")
  plot_learning_curves(history)


def calculate_confusion_matrix_and_accuracy(model, validation):
  all_labels = []
  all_predictions = []
  for data_batch, labels_batch in validation:
    predictions_batch = model.predict_on_batch(data_batch)
    all_labels.extend(np.argmax(labels_batch, axis=-1))
    all_predictions.extend(predictions_batch)

  predicted_classes = np.argmax(np.array(all_predictions), axis=-1) 

  cm = confusion_matrix(all_labels, predicted_classes)

  _, test_acc = model.evaluate(validation)

  return cm, test_acc

def visualize_confusion_matrix(cm):
  plt.figure(figsize=(10, 7))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title("Confusion Matrix")
  plt.savefig("plots/confusion_matrix.png")

def plot_learning_curves(history):
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history["accuracy"], label="Training Accuracy")
  plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.grid(True, linestyle="--", color="grey")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history.history["loss"], label="Training Loss")
  plt.plot(history.history["val_loss"], label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.grid(True, linestyle="--", color="grey")
  plt.legend()

  plt.tight_layout()
  plt.savefig("plots/learning_curves.png")
