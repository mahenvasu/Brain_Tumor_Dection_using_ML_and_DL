import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import joblib

# Data loading function (no masks)
def load_data(image_dir):
    images, labels = [], []
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        label = folder.lower()
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize to 224x224
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# label encoding (data preprocessing)
def encode_labels(labels):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    return y_encoded, encoder

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Load dataset
image_dir = r"dataset\image"
images, labels = load_data(image_dir)
labels_encoded, label_encoder = encode_labels(labels)

# Normalization (data preprocessing)
images = images / 255.0

# Train-test split (data preprocessing)
if len(images) > 1:
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
else:
    raise ValueError("Dataset must contain more than one image to split into training and testing sets.")

# Build CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Input shape changed to (224, 224, 3)
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

cnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
EPOCHS = 50
BATCH_SIZE = 32
history = cnn_model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop])

# Save model and label encoder
cnn_model.save('cnn_brain_tumor_model.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluation
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predictions
pred_probs = cnn_model.predict(X_test)
predictions = np.argmax(pred_probs, axis=1)

# Metrics
f1 = f1_score(y_test, predictions, average='weighted')
acc = accuracy_score(y_test, predictions)
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test, predictions))
plot_confusion_matrix(y_test, predictions)

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend()
plt.grid()
plt.show()
