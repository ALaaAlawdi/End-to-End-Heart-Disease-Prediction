import numpy as np
import pandas as pd
import pickle
import shap
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Load the RandomForestClassifier model
model_name = 'models/heart_disease_model_2023-09-29.pkl'  # Replace with the actual path to your model file
with open(model_name, 'rb') as model_file:
    MODEL = pickle.load(model_file)

# Define feature names (replace with your actual feature names)
feature_names = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
        'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
       'Asthma', 'KidneyDisease', 'SkinCancer']

def predict_hf(data: pd.DataFrame):
    # Ensure that column names match the model's feature names
    data_predict = data[feature_names]

    return MODEL.predict_proba(data_predict)[:, 1]

def get_shap_df(data: pd.DataFrame):
    # Ensure that column names match the model's feature names
    data_predict = data[feature_names]

    # Calculate SHAP values using the shap library
    explainer = shap.Explainer(MODEL)
    shap_values = explainer.shap_values(data_predict)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['AbsVal'] = np.abs(shap_df).mean(axis=0)  # Calculate the absolute SHAP value importance
    shap_df.sort_values('AbsVal', ascending=False, inplace=True)

    return shap_df

def plot_shap_values(data: pd.DataFrame):
    shap_df = get_shap_df(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=shap_df.index, y=shap_df['AbsVal']))
    fig.update_layout(title='Feature Importance')

    return fig.to_json()
