import streamlit as st
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from openai import OpenAI
import os
import zipfile
import urllib.request

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="OncoVision AI | Multi-Modal Diagnostics", layout="wide", page_icon="🧬")

# Initialize session state for the Chatbot context
if "medical_context" not in st.session_state:
    st.session_state.medical_context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# AUTO-DOWNLOAD & UNZIP MODELS (GitHub Limit Bypass)
# ==========================================
@st.cache_resource(show_spinner=False)
def prepare_models():
    # Check if ALL three models exist locally
    models_exist = (os.path.exists('mammogram_densenet.h5') and 
                    os.path.exists('breakhis_extratrees.pkl') and 
                    os.path.exists('metabric_rsf.pkl'))
    
    # If they are already unpacked, skip the download
    if models_exist:
        return
    
    # We'll save the downloaded file as 'all_models.zip' locally
    zip_path = "all_models.zip"
    
    # Download the ZIP from your GitHub Release
    if not os.path.exists(zip_path):
        st.info("Downloading Multi-Modal AI Models from cloud storage. This takes a minute...")
        urllib.request.urlretrieve("https://github.com/nithujega624-wq/breastcancer/releases/download/v1.1/metabric_rsf.zip", zip_path)
    
    # Extract the ZIP silently
    if os.path.exists(zip_path):
        with st.spinner("Unpacking AI Models..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")

prepare_models()

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    # Load Keras Model (compile=False saves memory)
    mammo_model = tf.keras.models.load_model('mammogram_densenet.h5', compile=False)
    # Load Sklearn Models
    histo_model = joblib.load('breakhis_extratrees.pkl')
    surv_model = joblib.load('metabric_rsf.pkl')
    return mammo_model, histo_model, surv_model

try:
    mammo_model, histo_model, surv_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models. Details: {e}")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("🧬 OncoVision AI")
    st.markdown("**Multi-Modal Diagnostic System**")
    st.markdown("---")
    page = st.radio("Select Analysis Module:", 
                    ["Mammography (DenseNet)", 
                     "Histopathology (ExtraTrees)", 
                     "Prognosis (Survival Forest)", 
                     "AI Clinical Assistant"])
    
    st.markdown("---")
    st.caption("🔒 Research Purpose Only. Not for clinical use. University of Colombo School of Computing (UCSC) Final Year Project.")

# ==========================================
# MODULE 1: MAMMOGRAPHY
# ==========================================
if page == "Mammography (DenseNet)":
    st.title("🩺 Mammogram Diagnostics & XAI")
    st.markdown("Upload a mammogram to detect malignancy. Explainable AI highlights the regions driving the prediction.")
    
    uploaded_file = st.file_uploader("Upload Mammogram Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and models_loaded:
        col1, col2 = st.columns(2)
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        h, w = img_raw.shape[:2]
        img_cropped = img_raw[int(h*0.18):h, int(w*0.20):w]
        img_padded = cv2.copyMakeBorder(img_cropped, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_padded)
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        input_arr = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)
        
        with st.spinner("Analyzing scan..."):
            pred = mammo_model.predict(input_arr)[0][0]
            confidence = pred if pred > 0.5 else 1 - pred
            diagnosis = "MALIGNANT" if pred > 0.5 else "BENIGN"
            
            st.session_state.medical_context += f"\nMammogram Analysis: Predicted {diagnosis} with {confidence*100:.2f}% confidence."
        
        with col1:
            st.subheader("Enhanced Scan (CLAHE)")
            st.image(img_resized, channels="RGB", use_container_width=True)
            
            if diagnosis == "MALIGNANT":
                st.error(f"**Diagnosis:** {diagnosis} ({confidence*100:.2f}% Confidence)")
            else:
                st.success(f"**Diagnosis:** {diagnosis} ({confidence*100:.2f}% Confidence)")

        with col2:
            st.subheader("AI Focus Area (Grad-CAM)")
            try:
                base_model = mammo_model.get_layer('densenet121')
                last_conv = base_model.get_layer('conv5_block16_concat')
                final_weights = mammo_model.layers[-1].get_weights()[0]
                
                intermediate_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv.output)
                feature_maps = intermediate_model.predict(input_arr, verbose=0)[0]
                
                cam = np.zeros(feature_maps.shape[0:2], dtype=np.float32)
                for i, w_val in enumerate(final_weights):
                    cam += w_val[0] * feature_maps[:, :, i]
                    
                cam = np.maximum(cam, 0)
                cam = cam / (np.max(cam) + 1e-10)
                cam = cv2.resize(cam, (224, 224))
                
                mask = np.zeros((224, 224), dtype=np.float32)
                cv2.circle(mask, (112, 112), 100, 1, -1)
                cam = cam * mask
                cam = cam / (np.max(cam) + 1e-10)
                
                h_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
                
                superimposed = (h_color * 0.4) + (img_resized * 0.6)
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                
                st.image(superimposed, channels="RGB", use_container_width=True)
            except Exception as e:
                st.warning("Could not generate Grad-CAM. Ensure the model architecture matches the extraction logic.")

# ==========================================
# MODULE 2: HISTOPATHOLOGY
# ==========================================
elif page == "Histopathology (ExtraTrees)":
    st.title("🔬 Histology Texture Analysis")
    st.markdown("Digital Pathologist Feature Extraction using Color Statistics & GLCM Textures.")
    
    uploaded_file = st.file_uploader("Upload Histopathology Slide", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and models_loaded:
        col1, col2 = st.columns([1, 2])
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (128, 128))
        
        with col1:
            st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Analyzed Slide", use_container_width=True)
        
        with col2:
            with st.spinner("Extracting GLCM & Color features..."):
                feats = []
                for i in range(3):
                    feats.append(np.mean(img_resized[:,:,i]))
                    feats.append(np.std(img_resized[:,:,i]))
                
                hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                feats.append(np.mean(hsv[:,:,1])) 
                feats.append(np.mean(hsv[:,:,2])) 
                
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                glcm = graycomatrix(gray, [1], [0, np.pi/4], 256, symmetric=True, normed=True)
                for prop in ['contrast', 'energy', 'correlation', 'homogeneity']:
                    feats.append(np.mean(graycoprops(glcm, prop)))
                
                features_arr = np.array([feats])
                
                pred = histo_model.predict(features_arr)[0]
                diagnosis = "Malignant" if pred == 1 else "Benign"
                
                st.session_state.medical_context += f"\nHistopathology Analysis: Predicted {diagnosis}."
                
                if diagnosis == "Malignant":
                    st.error(f"**Slide Classification:** {diagnosis}")
                else:
                    st.success(f"**Slide Classification:** {diagnosis}")
                
                st.subheader("Feature Contribution")
                importances = histo_model.feature_importances_
                feature_names = ["Mean B", "Std B", "Mean G", "Std G", "Mean R", "Std R", 
                                 "Saturation", "Brightness", "Contrast", "Energy", "Correlation", "Homogeneity"]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                indices = np.argsort(importances)
                ax.barh(range(len(indices)), importances[indices], color='#8b5cf6', align='center')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices], color='white' if st.get_option("theme.base") == "dark" else 'black')
                ax.set_xlabel("Relative Importance Score")
                st.pyplot(fig)

# ==========================================
# MODULE 3: PROGNOSIS
# ==========================================
elif page == "Prognosis (Survival Forest)":
    st.title("📈 Survival Prognosis (METABRIC)")
    st.markdown("Patient risk stratification using Random Survival Forests.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age at Diagnosis", 20.0, 90.0, 50.0)
        tumor_size = st.slider("Tumor Size (mm)", 1.0, 100.0, 20.0)
        tumor_stage = st.selectbox("Tumor Stage", [1.0, 2.0, 3.0, 4.0])
        mutation_count = st.number_input("Mutation Count", 1.0, 20.0, 5.0)
        
    with col2:
        npi = st.slider("Nottingham Prognostic Index", 2.0, 7.0, 4.0)
        chemo = st.selectbox("Chemotherapy (0=No, 1=Yes)", [0.0, 1.0])
        hormone = st.selectbox("Hormone Therapy (0=No, 1=Yes)", [0.0, 1.0])
        radio = st.selectbox("Radio Therapy (0=No, 1=Yes)", [0.0, 1.0])

    if st.button("Calculate Survival Risk", type="primary"):
        patient_data = np.array([[age, tumor_size, tumor_stage, mutation_count, npi, chemo, hormone, radio]])
        
        with st.spinner("Calculating Risk Trajectory..."):
            chf_funcs = surv_model.predict_cumulative_hazard_function(patient_data)
            risk_score = chf_funcs[0](120) 
            
            st.session_state.medical_context += f"\nPrognosis Analysis: Patient has a 10-year cumulative hazard score of {risk_score:.2f} based on age {age}, tumor size {tumor_size}."
            
            st.metric("10-Year Cumulative Risk Score", f"{risk_score:.2f}")
            st.info("Higher scores indicate higher risk. Switch to the AI Assistant tab to interpret these results.")

# ==========================================
# MODULE 4: OPENAI CLINICAL ASSISTANT
# ==========================================
elif page == "AI Clinical Assistant":
    st.title("🤖 OncoBot Clinical Assistant")
    st.markdown("Discuss the generated diagnostic and prognostic results with the AI.")
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("⚠️ OpenAI API Key not found in Streamlit Secrets. Please configure it in your Streamlit Cloud deployment settings.")
        api_key = None

    if st.session_state.medical_context == "":
        st.info("💡 Please run a diagnosis (Mammogram, Histology, or Prognosis) first so I have data to analyze.")
    else:
        with st.expander("View Current Patient Context"):
            st.text(st.session_state.medical_context)
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the patient's results..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    system_prompt = f"You are an expert oncologist AI assistant. Here is the patient's data from our deep learning pipeline:\n{st.session_state.medical_context}\nAnswer the user's questions based strictly on this context. Be professional and concise."
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": system_prompt}] + 
                                 [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    )
                    
                    bot_reply = response.choices[0].message.content
                    with st.chat_message("assistant"):
                        st.markdown(bot_reply)
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")
