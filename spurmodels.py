import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from keras.models import load_model 

# --- 1. Mixed Model Loading ---
@st.cache_resource
def load_models():
    """Loads .pkl for ML and .h5 for DL into memory."""
    try:
        models = {
            "XGBoost": joblib.load("xgb.pkl"),
            "Random_Forest": joblib.load("rf.pkl"),
            "LSTM": load_model("lstm.h5", compile=False), 
            "GRU": load_model("gru.h5", compile=False)
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}. Check your filenames and directory.")
        st.stop()
models = load_models()

# --- 2. Data Processing Pipeline ---
def apply_sg_filter(data_series, window_length=5, polyorder=2):
    """Applies Savitzky-Golay filter for Phase 0.1 data rectification."""
    clean_series = data_series.replace([np.inf, -np.inf], np.nan)
    if clean_series.isna().any():
        clean_series = clean_series.interpolate(method='linear').bfill().ffill()
        
    if len(clean_series) < window_length:
        return clean_series 
        
    return savgol_filter(clean_series, window_length, polyorder)

def predict_and_evaluate(input_data, y_true=None):
    """
    Routes 2D data to ML models, reshapes to 3D for DL models.
    """
    # 1. Prepare 2D data for XGBoost and Random Forest
    X_2D = input_data.values 
    
    # 2. Prepare 3D data for LSTM and GRU [samples, time_steps, features]
    X_3D = X_2D.reshape((X_2D.shape[0], 1, X_2D.shape[1]))
    
    # 3. Generate Predictions (flatten Keras outputs to match 1D array)
    preds = {
        "XGBoost": models["XGBoost"].predict(X_2D),
        "Random_Forest": models["Random_Forest"].predict(X_2D),
        "LSTM": models["LSTM"].predict(X_3D, verbose=0).flatten(),
        "GRU": models["GRU"].predict(X_3D, verbose=0).flatten()
    }
    
    results = {}
    can_calculate_r2 = y_true is not None and len(y_true) > 1
    
    # 4. Evaluate Performance
    for model_name, prediction in preds.items():
        if can_calculate_r2:
            current_r2 = r2_score(y_true, prediction)
        else:
            current_r2 = None
            
        results[model_name] = {
            "prediction": prediction,
            "r2": current_r2
        }
    
    # 5. Route Output Based on Available Metrics
    if can_calculate_r2:
        best_model_name = max(results, key=lambda k: results[k]["r2"])
        best_results = results[best_model_name]
        return best_model_name, best_results["prediction"], best_results["r2"], results, "Dynamic (Calculated)"
    else:
        return None, None, None, results, "Prediction Only (No Target Provided)"

# --- 3. Excel Export Utility ---
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Model_Predictions')
    processed_data = output.getvalue()
    return processed_data

# --- 4. User Interface ---
st.set_page_config(page_title="Hydraulic Scour Predictor", layout="wide")
st.title("Scour Depth Prediction Engine")
st.markdown("XGBoost, Random Forest, LSTM, and GRU inference engine. Dynamic routing applied when observed target data is available.")

input_method = st.sidebar.radio("Select Input Method:", ("Manual Entry", "Upload Excel File"))

features = ["Fr", "Re", "v/vc", "L1/d", "L2/d", "H/d", "T/d", "(B-L1)/d", "tv/d"]

if input_method == "Manual Entry":
    st.subheader("Enter Parameters Manually")
    st.info("Manual entries do not contain ground truth targets. R² cannot be calculated. The engine will output predictions from all four models simultaneously.")
    
    cols = st.columns(3)
    input_vals = {}
    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_vals[feature] = st.number_input(feature, value=0.0, format="%.4f")
            
    if st.button("Run Prediction"):
        df_input = pd.DataFrame([input_vals])
        best_model, pred, best_r2, all_results, eval_method = predict_and_evaluate(df_input)
        
        st.write("### Model Predictions")
        breakdown_df = pd.DataFrame([
            {"Model": m, "Predicted Scour Depth": all_results[m]["prediction"][0]}
            for m in all_results
        ])
        st.dataframe(breakdown_df, use_container_width=True)

elif input_method == "Upload Excel File":
    st.subheader("Batch Process via Excel")
    uploaded_file = st.file_uploader("Upload Excel file containing parameter columns", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in uploaded file: {missing_cols}")
        else:
            st.write("Raw Data Preview:", df.head())
            
            possible_target_cols = [col for col in df.columns if col not in features]
            st.markdown("---")
            st.write("**Evaluate Model Performance (Optional)**")
            target_col = st.selectbox(
                "If your file contains actual measured scour depths, select the column to calculate R² and automatically route to the best model:", 
                ["-- No Target Column (Predict Only) --"] + possible_target_cols
            )
            st.markdown("---")
            
            if st.button("Process Batch Data"):
                with st.spinner("Applying SG Filter and running inference..."):
                    
                    # Apply Phase 0.1 Rectification
                    for col in features:
                        df[f"{col}_filtered"] = apply_sg_filter(df[col])
                    
                    # Extract ground truth if selected
                    if target_col != "-- No Target Column (Predict Only) --" and len(df) > 1:
                        y_true_data = df[target_col].values
                    else:
                        y_true_data = None
                        
                    # Evaluate
                    best_model, preds, best_r2, all_results, eval_method = predict_and_evaluate(
                        df[[f"{col}_filtered" for col in features]], 
                        y_true=y_true_data
                    )
                    
                    if best_model:
                        # Dynamic Routing Path (Ground Truth Available)
                        df["Optimal_Model_Used"] = best_model
                        df["Predicted_Scour_Depth"] = preds
                        st.success(f"Batch processing complete. Optimal engine dynamically selected: **{best_model}**")
                        
                        st.write("### Model Metric Summary")
                        summary_df = pd.DataFrame([
                            {"Model": m, "Dynamic R² Score": all_results[m]["r2"]}
                            for m in all_results
                        ])
                        st.dataframe(summary_df)
                    else:
                        # Prediction-Only Path (No Ground Truth)
                        for m in all_results:
                            df[f"{m}_Prediction"] = all_results[m]["prediction"]
                        st.success("Batch processing complete. Predictions appended for all four models.")
                    
                    # Drop the filtered columns before export for a cleaner sheet
                    export_df = df.drop(columns=[f"{col}_filtered" for col in features])
                    
                    st.dataframe(export_df)
                    
                    excel_data = convert_df_to_excel(export_df)
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=excel_data,
                        file_name='scour_predictions_output.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
