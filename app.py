import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

# Load models
cnn_model = load_model('models/cnn_brain_tumor_model.h5')
mobilenet_model = load_model('models/mobilenetv2_brain_tumor_model.h5')
ensemble_model = load_model('models/ensemble_brain_tumor_model_20250423_084058.h5')
label_encoder = joblib.load('models/label_encoder.pkl')

# Label to name and treatment mapping
tumor_info = {
    '0': {'name': 'No Tumor', 'message': 'Your brain scan looks normal! 🧘‍♂️ Keep up the healthy lifestyle! 💚', 'treatment': None},
    '1': {'name': 'Glioma Tumor', 'message': 'Detected signs of Glioma. 💪 Early detection saves lives!', 'treatment': ['Radiation Therapy', 'Surgical Removal', 'Chemotherapy']},
    '2': {'name': 'Meningioma Tumor', 'message': 'Meningioma signs detected. 🌟 Stay strong – medical support is excellent.', 'treatment': ['Surgical Removal', 'Observation', 'Radiation']},
    '3': {'name': 'Pituitary Tumor', 'message': 'Pituitary tumor found. 💡 Most types are curable with proper care!', 'treatment': ['Hormonal Therapy', 'Surgery', 'Radiation Therapy']}
}

# Preprocess image
def load_and_preprocess_image(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict tumor type
def predict(img_file):
    img_array = load_and_preprocess_image(img_file)
    cnn_preds = cnn_model.predict(img_array)
    mobilenet_preds = mobilenet_model.predict(img_array)
    combined_preds = np.concatenate([cnn_preds, mobilenet_preds], axis=1)
    ensemble_preds = ensemble_model.predict(combined_preds)
    final_pred = np.argmax(ensemble_preds, axis=1)
    predicted_label = label_encoder.inverse_transform(final_pred)[0]
    return str(predicted_label), tumor_info[str(predicted_label)]

# ----------------------- UI ----------------------------

st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")

st.markdown("""
    <style>
        .main-title {
            font-size: 3em;
            font-weight: bold;
            color: #4A00E0;
            text-align: center;
        }
        .subtext {
            text-align: center;
            font-size: 1.2em;
            color: #555;
        }
        .result-box {
            background: linear-gradient(to right, #4facfe, #00f2fe);
            padding: 20px;
            border-radius: 20px;
            color: white;
            font-size: 1.5em;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .footer {
            text-align: center;
            color: #aaa;
            font-size: 0.9em;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload an MRI scan to identify the tumor type</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a brain tumor image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    
    with st.spinner('🔍 Analyzing the MRI...'):
        tumor_id, tumor_details = predict(uploaded_file)

    # Main prediction result
    st.markdown(f"""
        <div class="result-box">
            🧾 <strong>Predicted Tumor Type:</strong> {tumor_id} <br>
            🧬 <strong>Tumor Name:</strong> {tumor_details['name']}
        </div>
    """, unsafe_allow_html=True)

    # Sidebar message
    st.sidebar.title("🩺 Medical Insight")
    st.sidebar.info(tumor_details['message'])

    # If tumor, show cure suggestions
    if tumor_details['treatment']:
        st.sidebar.subheader("💊 Suggested Treatment Options")
        for option in tumor_details['treatment']:
            st.sidebar.markdown(f"- {option}")
    else:
        st.sidebar.success("🎉 No treatment needed!")

st.markdown('<div class="footer">Made with ❤️ by Vasavan </div>', unsafe_allow_html=True)
