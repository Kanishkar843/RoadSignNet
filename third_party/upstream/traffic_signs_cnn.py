import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set paths (replace with actual paths where you store the data)
DATA_DIR = 'data/GTSRB/Final_Training/Images/'
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43

def load_gtsrb_data(data_dir):
    # This is a placeholder function.
    # You need to implement reading images and labels from the GTSRB dataset.
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, format(class_id, '05d'))
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith('.ppm'):
                img_path = os.path.join(class_dir, file)
                img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
                images.append(np.array(img))
                labels.append(class_id)
    X = np.array(images)
    y = np.array(labels)
    return X, y

print("Loading data...")
X, y = load_gtsrb_data(DATA_DIR)
X = X.astype('float32') / 255.0
y = to_categorical(y, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building model...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print("Training...")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Save sample training curve
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/training_accuracy.png')

# Evaluate and show sample predictions
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

# Show some sample predictions
preds = model.predict(X_test)
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i])
    plt.title(f"Pred: {np.argmax(preds[i])}\nTrue: {np.argmax(y_test[i])}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('images/sample_predictions.png')
plt.show()
