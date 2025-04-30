import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    return img

def ensemble_segment(img, model1, model2):
    input_img = cv2.resize(img, (224, 224)) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred1 = model1.predict(input_img)[0]
    pred2 = model2.predict(input_img)[0]

    ensemble_pred = (pred1 + pred2) / 2
    mask = (ensemble_pred > 0.5).astype(np.uint8)
    mask = np.squeeze(mask) * 255
    return mask

def extract_features(img):
    return img.flatten().reshape(1, -1)

def classify_ensemble(features, cnn_model, rf_model, xgb_model, encoder):
    cnn_pred = cnn_model.predict(features.reshape(-1, 224, 224, 3) / 255.0)
    rf_pred = rf_model.predict_proba(features)
    xgb_pred = xgb_model.predict_proba(features)

    avg_pred = (cnn_pred + rf_pred + xgb_pred) / 3
    class_index = np.argmax(avg_pred)
    confidence = avg_pred[0][class_index]
    label = encoder.inverse_transform([class_index])[0]
    return label, confidence

def mask_area_percentage(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    tumor_pixels = np.sum(mask > 0)
    return (tumor_pixels / total_pixels) * 100
