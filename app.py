import streamlit as st
import pandas as pd
import numpy as np
from sales_prediction import SalesPredictor
import traceback
import io

st.set_page_config(page_title="Sales Prediction Dashboard", page_icon="📊")
st.title("📊 Sales Prediction Dashboard")

# Add sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This dashboard uses ML models to predict sales based on quantity and unit price.")
    st.write("**Models:** Linear Regression & Random Forest")
    st.write("**Metrics:** R² Score & RMSE")

# Initialize predictor
@st.cache_resource
def load_model():
    try:
        predictor = SalesPredictor()
        # Train with sample data
        sample_data = pd.DataFrame({
            'quantity': [10, 25, 5, 30, 15, 40, 8, 22, 35, 12, 18, 45, 7, 28, 33],
            'unit_price': [100, 50, 200, 75, 120, 60, 180, 90, 45, 150, 110, 55, 170, 85, 65]
        })
        results = predictor.train(sample_data)
        return predictor, results, None
    except Exception as e:
        return None, None, str(e)

predictor, model_results, error = load_model()

if error:
    st.error(f"Failed to load model: {error}")
    st.stop()

# Display model performance
st.subheader("Model Performance")
try:
    col1, col2 = st.columns(2)
    with col1:
        rf_r2 = model_results['rf_r2']
        rf_rmse = model_results['rf_rmse']
        st.metric("Random Forest R²", f"{rf_r2:.3f}" if not np.isnan(rf_r2) else "N/A")
        st.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
    with col2:
        lr_r2 = model_results['linear_r2']
        lr_rmse = model_results['linear_rmse']
        st.metric("Linear Regression R²", f"{lr_r2:.3f}" if not np.isnan(lr_r2) else "N/A")
        st.metric("Linear Regression RMSE", f"{lr_rmse:.2f}")
    
    # Feature importance
    st.subheader("Feature Analysis")
    st.info(f"Most influential feature: **{model_results['feature_importance']['most_influential']}**")
    feature_df = pd.DataFrame({
        'Feature': ['quantity', 'unit_price'],
        'Linear Coef': list(model_results['feature_importance']['linear_coef'].values()),
        'RF Importance': list(model_results['feature_importance']['rf_importance'].values())
    })
    st.dataframe(feature_df)
except Exception as e:
    st.error(f"Error displaying model metrics: {e}")

# Prediction interface
st.subheader("Sales Prediction")
try:
    quantity = st.number_input("Quantity", min_value=1, max_value=10000, value=20)
    unit_price = st.number_input("Unit Price ($)", min_value=0.01, max_value=10000.0, value=80.0, format="%.2f")
    model_type = st.selectbox("Model", ["rf", "lr"], format_func=lambda x: "Random Forest" if x == "rf" else "Linear Regression")
    
    if st.button("Predict Sales"):
        try:
            with st.spinner("Making prediction..."):
                prediction = predictor.predict(quantity, unit_price, model_type)
                st.success(f"Predicted Total Sales: ${prediction:.2f}")
                
                # Alert system
                if prediction > 2000:
                    st.warning("⚠️ High sales prediction - Consider inventory alert")
                elif prediction < 500:
                    st.info("ℹ️ Low sales prediction - Review pricing strategy")
                    
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
except Exception as e:
    st.error(f"Error in prediction interface: {e}")

# File upload for custom data
st.subheader("Upload Custom Data")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    try:
        custom_data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(custom_data.head())
        
        if st.button("Retrain Model"):
            with st.spinner("Retraining model..."):
                new_predictor = SalesPredictor()
                new_results = new_predictor.train(custom_data)
                st.success("Model retrained successfully!")
                st.json(new_results)
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")