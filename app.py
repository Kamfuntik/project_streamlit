# app.py
import os
import json
import joblib
from datetime import datetime, date, time as pytime
import pandas as pd
import numpy as np
import streamlit as st

PIPELINE_PATH = os.path.join("models", "best_pipeline.pkl")
SCHEMA_PATH = "model_schema.json"
METADATA_PATH = os.path.join("models", "model_info.json")  

GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID")  
GDRIVE_MODEL_PATH = PIPELINE_PATH

st.set_page_config(page_title="Taxi Wait Time Predictor", layout="centered")

def download_from_gdrive(file_id: str, out_path: str):
    try:
        import gdown
    except Exception:
        st.info("gdown not installed. Installing gdown to download model from Google Drive...")
        os.system("pip install gdown -q")
        import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)

@st.cache_resource(show_spinner=False)
def load_artifacts(pipeline_path=PIPELINE_PATH, schema_path=SCHEMA_PATH):
   
    if not os.path.exists(pipeline_path):
        if GDRIVE_FILE_ID:
            os.makedirs(os.path.dirname(pipeline_path) or ".", exist_ok=True)
            download_from_gdrive(GDRIVE_FILE_ID, pipeline_path)
        else:
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}. Put your pipeline there or set GDRIVE_FILE_ID.")
    pipeline = joblib.load(pipeline_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)
   
    meta = None
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
    except Exception:
        meta = None
    return pipeline, schema, meta

def datetime_to_derived(dt):
   
    hour = int(dt.hour)
    dow = int(dt.weekday())
    is_weekend = 1 if dow >= 5 else 0
    month = int(dt.month)
    return hour, dow, is_weekend, month

def build_model_input_df(schema, user_raw):
    """
    schema: loaded model_schema.json
    user_raw: dict of raw user inputs keyed by raw_input_columns
    returns: DataFrame with columns schema['model_input_columns']
    """
    model_cols = schema['model_input_columns']
    dt_col = schema.get('datetime_col')
    numeric_medians = schema.get('numeric_medians', {})
    numeric_feats = set(schema.get('numeric_features', []))

   
    row = {}
    for c in model_cols:
        if c in numeric_medians:
            row[c] = numeric_medians[c]
        elif c in numeric_feats:
            row[c] = 0.0
        else:
            row[c] = ""

    
    for raw_col, val in user_raw.items():
        if raw_col == dt_col:
            if pd.isna(val) or val is None or val == "":
                continue
            dt_val = pd.to_datetime(val)
            hour, dow, is_weekend, month = datetime_to_derived(dt_val)
            for k, v in {
                f"{dt_col}_hour": hour,
                f"{dt_col}_dow": dow,
                f"{dt_col}_is_weekend": is_weekend,
                f"{dt_col}_month": month
            }.items():
                if k in row:
                    row[k] = v
            
            if dt_col in row:
                row[dt_col] = dt_val
        else:
            
            if raw_col in row:
                row[raw_col] = val
            else:
               
                pass

    X = pd.DataFrame([row], columns=model_cols)
    return X


try:
    pipeline, schema, metadata = load_artifacts()
except Exception as e:
    st.error("Could not load model artifacts.")
    st.exception(e)
    st.stop()

# ---------- UI ----------
st.title("🚕 Taxi Wait Time Predictor (Demo)")

with st.expander("Model summary / info", expanded=False):
    if metadata:
        st.write("**Model metrics (from training)**")
        st.write(metadata)
    else:
        st.write("No metadata (model_info.json) available. If you saved it during training it will show here.")
    st.write("**Model pipeline file:**", PIPELINE_PATH)
    st.write("**Model schema keys:**", list(schema.keys()))

st.markdown("---")
st.markdown("### Enter trip/request details")
st.write("Defaults are set to training medians where available. Use the **Load sample defaults** button to quickly fill defaults.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    load_sample = st.button("Load sample defaults")
    st.write("Model file:", PIPELINE_PATH)
    st.write("Schema file:", SCHEMA_PATH)

# Render inputs dynamically
raw_cols = schema['raw_input_columns']
dt_col = schema.get('datetime_col')
numeric_medians = schema.get('numeric_medians', {})
categories = schema.get('categories_sample', {})
numeric_feats = set(schema.get('numeric_features', []))
categorical_feats = set(schema.get('categorical_features', []))

user_raw = {}
cols_iter = st.columns(2)  # 2-column layout for neatness
col_idx = 0
for col_name in raw_cols:
    target_col = cols_iter[col_idx % 2]
    with target_col:
        if col_name == dt_col:
            # date & time inputs side-by-side
            default_dt = pd.to_datetime(datetime.now())
            d = st.date_input(f"{col_name} (date)", value=default_dt.date())
            t = st.time_input(f"{col_name} (time)", value=pytime(default_dt.hour, default_dt.minute))
            combined = pd.to_datetime(datetime.combine(d, t))
            user_raw[col_name] = combined if not load_sample else combined
        elif col_name in numeric_feats:
            default_val = numeric_medians.get(col_name, 0.0)
            if load_sample:
                val = default_val
            else:
                val = st.number_input(col_name, value=float(default_val))
            user_raw[col_name] = float(val)
        elif col_name in categorical_feats:
            opts = categories.get(col_name, [])
            if opts:
                # keep selectbox readable even with many options by showing first 100
                opt_list = opts if len(opts) <= 200 else opts[:200]
                try:
                    selection = st.selectbox(col_name, opt_list)
                except Exception:
                    selection = st.text_input(col_name, value=str(opt_list[0] if opt_list else ""))
                user_raw[col_name] = selection
            else:
                user_raw[col_name] = st.text_input(col_name, value="")
        else:
            user_raw[col_name] = st.text_input(col_name, value="")

    col_idx += 1

st.markdown("---")
if st.button("Predict"):
    try:
        X_input = build_model_input_df(schema, user_raw)
        st.subheader("Model input preview (first row)")
        st.dataframe(X_input.T)
        pred = pipeline.predict(X_input)[0]
        st.metric("Predicted wait time", f"{round(float(pred), 2)} (same units as training target)")
    except Exception as e:
        st.error("Prediction failed: see error below.")
        st.exception(e)
        st.write("Common causes: pipeline/schema mismatch, missing columns in model_input_columns, or custom transformers not available.")

st.markdown("---")
st.caption("Built with Streamlit — this is a prototype UI intended for demos and presentations.")
