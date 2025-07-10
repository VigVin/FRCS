# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Set page configuration
st.set_page_config(
    page_title="Concrete Properties Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['total_aggregate'] = X['coarse_aggregate'] + X['fine_aggregate']
        X['cement_aggregate_ratio'] = X['cement'] / X['total_aggregate']
        X['water_content'] = X['cement'] * X['water_cement_ratio']
        return X

# Load models
@st.cache_resource
def load_models():
    try:
        fire_model = joblib.load('fire_resistance_model.pkl')
        strength_model = joblib.load('compressive_strength_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return fire_model, strength_model, preprocessor
    except:
        st.error("Model files not found! Please ensure you have the trained models saved as:")
        st.code("fire_resistance_model.pkl\ncompressive_strength_model.pkl\npreprocessor.pkl")
        st.stop()

fire_model, strength_model, preprocessor = load_models()

# App header
st.title("🏗️ Concrete Properties Prediction")
st.markdown("""
Predict fire resistance and compressive strength of concrete composites based on composition parameters.
""")

# Sidebar for input parameters
with st.sidebar:
    st.header("⚙️ Concrete Composition Parameters")
    
    cement = st.number_input("Cement (kg/m³)", min_value=200, max_value=1500, value=425, step=10)
    coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0, max_value=1500, value=1113, step=10)
    fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0, max_value=1500, value=682, step=10)
    wc_ratio = st.slider("Water-Cement Ratio", min_value=0.1, max_value=0.7, value=0.36, step=0.01)
    
    st.subheader("Advanced Parameters")
    col1, col2 = st.columns(2)
    with col1:
        density = st.number_input("Density (kg/m³)", min_value=1800, max_value=3000, value=2400, step=10)
        thickness = st.number_input("Thickness (mm)", min_value=10, max_value=500, value=180, step=10)
    with col2:
        agg_type = st.selectbox("Aggregate Type", 
                              options=["Siliceous (0)", "Carbonate (1)", "Lightweight (2)"],
                              index=0)
        fiber_reinf = st.selectbox("Fiber Reinforcement", 
                                options=["None (0)", "Polypropylene (1)", "Steel (2)", "Both (3)"],
                                index=1)
    
    silica_fume = st.slider("Silica Fume (% of cement)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)
    
    predict_btn = st.button("Predict Concrete Properties", use_container_width=True)

# Convert categorical inputs to numeric
agg_type_map = {"Siliceous (0)": 0, "Carbonate (1)": 1, "Lightweight (2)": 2}
fiber_reinf_map = {
    "None (0)": 0, 
    "Polypropylene (1)": 1, 
    "Steel (2)": 2, 
    "Both (3)": 3
}

# Prepare input data
input_data = {
    'cement': cement,
    'coarse_aggregate': coarse_agg,
    'fine_aggregate': fine_agg,
    'water_cement_ratio': wc_ratio,
    'aggregate_type': agg_type_map[agg_type],
    'density': density,
    'thickness': thickness,
    'fiber_reinforcement': fiber_reinf_map[fiber_reinf],
    'silica_fume': silica_fume
}

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df.style.format("{:.2f}"), height=300)
    
    st.markdown("""
    **Aggregate Type Codes:**
    - 0 = Siliceous
    - 1 = Carbonate
    - 2 = Lightweight
    
    **Fiber Reinforcement Codes:**
    - 0 = None
    - 1 = Polypropylene fiber
    - 2 = Steel fiber
    - 3 = Both
    """)

with col2:
    st.subheader("Prediction Results")
    
    if predict_btn:
        with st.spinner("Predicting concrete properties..."):
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Apply feature engineering
            feature_engineer = FeatureEngineer()
            engineered_df = feature_engineer.transform(input_df)
            
            # Make predictions
            fire_resistance = fire_model.predict(engineered_df)[0]
            compressive_strength = strength_model.predict(engineered_df)[0]
            
        # Display results
        st.success("Prediction completed successfully!")
        
        # Fire resistance visualization
        st.metric("Fire Resistance", f"{fire_resistance:.1f} minutes")
        fire_progress = min(100, fire_resistance / 3)
        st.progress(fire_progress/100, text=f"Rating: {'Excellent' if fire_resistance > 120 else 'Good' if fire_resistance > 60 else 'Fair'}")
        
        # Compressive strength visualization
        st.metric("Compressive Strength", f"{compressive_strength:.1f} MPa")
        strength_progress = min(100, compressive_strength / 5)
        st.progress(strength_progress/100, text=f"Rating: {'Excellent' if compressive_strength > 60 else 'Good' if compressive_strength > 40 else 'Fair'}")
        
        # Interpretation
        st.subheader("Interpretation")
        if fire_resistance > 120:
            st.info("🔥 Excellent fire resistance - Suitable for high-temperature applications")
        elif fire_resistance > 60:
            st.info("🔥 Good fire resistance - Suitable for most building applications")
        else:
            st.warning("🔥 Fair fire resistance - Consider adding fire-resistant additives")
            
        if compressive_strength > 60:
            st.info("💪 High-strength concrete - Suitable for structural applications")
        elif compressive_strength > 40:
            st.info("💪 Medium-strength concrete - Suitable for most construction purposes")
        else:
            st.warning("💪 Low-strength concrete - Consider optimizing mix design")
    else:
        st.info("Click the 'Predict' button to see results")

# Model information
st.divider()
st.subheader("About the Prediction Model")
st.markdown("""
This application uses an optimized machine learning framework to predict concrete properties:

- **Model Type:** Ensemble of Random Forest and Gradient Boosting models
- **Training Data:** Comprehensive dataset of concrete compositions and test results
- **Features:** 9 input parameters describing concrete composition
- **Targets:** Fire resistance (minutes) and Compressive Strength (MPa)

The model has been validated with R² scores above 0.85 for both targets, ensuring reliable predictions.
""")

# How to use
with st.expander("Usage Instructions"):
    st.markdown("""
    1. Adjust the concrete composition parameters in the sidebar
    2. Click the **Predict Concrete Properties** button
    3. View predictions in the main panel
    4. Results include:
        - Fire resistance in minutes
        - Compressive strength in MPa
        - Quality ratings
        - Engineering recommendations
    """)

# Footer
st.divider()
st.caption("Developed by Concrete AI Research Team | Using optimized ML framework")