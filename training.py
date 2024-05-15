from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization, Dropout


def train_model(train, validation):
  model = create_model()
  history = History()
  checkpoint = ModelCheckpoint("detection_model.keras", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
  early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

  model.fit(train, validation_data=validation, epochs=20, batch_size=16, callbacks=[history, checkpoint, early_stopping])

  return model, history


def create_model():
  model = Sequential([
    Input(shape=(48, 48, 3)),
    Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation="relu", kernel_initializer="he_uniform"),
    Dense(7, activation="sigmoid")
  ])

  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

  return model
