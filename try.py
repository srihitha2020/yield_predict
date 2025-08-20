import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ===============================
# Load & preprocess data
# ===============================
FILE_PATH = r"1Y02_Data.xlsx"

@st.cache_data
def load_data():
    all_sheets = pd.read_excel(FILE_PATH, sheet_name=None)
    df = pd.concat(all_sheets.values(), ignore_index=True)

    # Drop useless columns
    drop_cols = ["Unnamed: 6", "Unnamed: 7", "Unnamed: 12",
                 "4AS3BKS", "recordstamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

df = load_data()
st.write(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

# Target & top features
target = "BFOUT"
top_features = ["BFIN", "TALLYLENGTH", "TALLYWIDTH", "MATERIAL",
                "MATERIALTHICKNESS", "MATERIALSPECIE", "TALLYGRADE"]

df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target]

# Handle datetime columns
for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
    X[col + "_year"] = X[col].dt.year
    X[col + "_month"] = X[col].dt.month
    X[col + "_day"] = X[col].dt.day
    X[col + "_dayofweek"] = X[col].dt.dayofweek
    X.drop(columns=[col], inplace=True)

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# Streamlit User Input (ONLY top features)
# ===============================
st.subheader("🔮 Predict BFOUT")

input_data = {}
for col in top_features:
    if col in label_encoders:  # categorical
        options = list(label_encoders[col].classes_)
        input_val = st.selectbox(f"Select {col}", options)
        input_data[col] = label_encoders[col].transform([input_val])[0]
    else:  # numeric
        input_val = st.number_input(
            f"Enter {col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
        input_data[col] = input_val

# Fill missing columns with defaults
for col in X.columns:
    if col not in input_data:
        if col in label_encoders:
            input_data[col] = 0  # default category
        else:
            input_data[col] = float(X[col].mean())  # mean for numeric

# ✅ Align order with training features
input_df = pd.DataFrame([input_data])[X.columns]

if st.button("Predict BFOUT"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted BFOUT: {prediction:.2f}")

