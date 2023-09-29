import pandas as pd
from flask import Blueprint, request
from services.model_utils import predict_hf, plot_shap_values

# Create a Flask Blueprint named 'bp1' with a URL prefix '/main'
bp1 = Blueprint('main', __name__, url_prefix='/main')

# Define a route for making predictions via POST request
@bp1.route('/api/make_prediction', methods=['POST'])
def make_prediction():

    # Extract form data as a DataFrame (assuming the form data contains feature values)
    form_df: pd.DataFrame = pd.DataFrame(request.form, index=[0])

    # Use the 'predict_hf' function to make a prediction on the provided data
    pred = predict_hf(form_df)
    
    # Extract the probability of class 1 (assuming it's a binary classification problem)
    pred_class_1 = pred[0]

    # Generate a plot using the 'plot_shap_values' function
    plot = plot_shap_values(form_df)

    # Return the prediction and the plot as JSON response
    return {'pred': round(pred_class_1, 2), 'plot': plot}
