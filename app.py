import streamlit as st
import pickle
import numpy as np

# ==========================================
# 1. PAGE SETUP & MODEL LOADING
# ==========================================
st.set_page_config(
    page_title="Student Placement Predictor", 
    page_icon="🎓", 
    layout="centered"
)

# Cache the model so it doesn't reload and slow down on every slider click
@st.cache_resource
def load_ml_artifacts():
    # Load your trained model file
    model = pickle.load(open("placement_model.pkl", "rb"))
    
    # ⚠️ NOTE: If you used a Scaler (StandardScaler) in your notebook, 
    # save it as 'scaler.pkl' and uncomment the line below:
    # scaler = pickle.load(open("scaler.pkl", "rb"))
    scaler = None
    
    return model, scaler

try:
    model, scaler = load_ml_artifacts()
except FileNotFoundError as e:
    st.error("📁 Missing required model files! Please ensure 'placement_model.pkl' is uploaded to your root GitHub folder.")
    st.stop()

# ==========================================
# 2. USER INTERFACE (WIDGETS)
# ==========================================
st.title("🎓 Student Placement Predictor")
st.write("Adjust the student metrics below to compute real-time placement probabilities.")
st.markdown("---")

# Layout using columns for a polished look
col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("Cumulative GPA (CGPA)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    profile_score = st.slider("Profile Score", min_value=0, max_value=100, value=65, step=1)

with col2:
    iq = st.slider("IQ Score", min_value=50, max_value=150, value=100, step=1)

st.markdown("---")

# ==========================================
# 3. PREDICTION & INFERENCE ENGINE
# ==========================================
if st.button("Predict Placement Status", type="primary"):
    
    # CRITICAL: Features must match the exact sequence your model was trained on!
    # Array order: [cgpa, iq, profile_score]
    raw_features = np.array([[cgpa, iq, profile_score]])
    
    # Transform data if a scaler artifact exists
    if scaler is not None:
        input_transformed = scaler.transform(raw_features)
    else:
        input_transformed = raw_features

    try:
        # 1. Fix: Generate the explicit prediction from the current slider inputs
        prediction = model.predict(input_transformed)
        
        # 2. Fix: Calculate probabilities 
        probability = model.predict_proba(input_transformed)
        placement_prob = probability[0][1] * 100  # Confidence percentage for Class 1 (Placed)

        # ==========================================
        # 4. RESULTS DISPLAY
        # ==========================================
        st.subheader("Analysis Metrics")
        
        # Dynamic outcome cards based on the explicit prediction check
        if prediction[0] == 1:
            st.success(f"🎉 **Placement Prediction: Likely to be Placed!**")
        else:
            st.error(f"❌ **Placement Prediction: Unlikely to be Placed.**")
            
        # Metric output display
        st.metric(label="Placement Confidence Probability", value=f"{placement_prob:.2f}%")
        st.progress(int(placement_prob))

    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        st.info("💡 Troubleshooting tip: Verify that your saved '.pkl' model matches the shape of your 3 slider variables.")
