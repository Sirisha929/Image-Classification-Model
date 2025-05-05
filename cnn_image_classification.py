# 📦 Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 🧠 Load CIFAR-10 dataset (already labeled)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 📏 Normalize pixel values (0 to 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 👀 Check the shape of the training data
print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)

# 🏗️ Build the CNN model
model = models.Sequential()

# 🧱 First layer: Convolution + Pooling
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))

# 🧱 Second layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# 🧱 Third layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# 🎯 Flatten the output + Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # 10 output classes

# 🛠️ Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ⏳ Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# ✅ Check model performance on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

# 📈 Plot training progress
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Save the trained model (optional)
model.save("cnn_model.h5")
