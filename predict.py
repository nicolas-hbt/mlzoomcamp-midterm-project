import pickle
import numpy as np
import pandas as pd
import train
from flask import Flask, request, jsonify


app = Flask("bike_rental_prediction")

with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input and returns model predictions.
    """
    # Get JSON data from the request
    json_data = request.get_json()

    # --- Create DataFrame from a *list* of dicts ---
    # This tells pandas to treat the dict as a single row
    X_raw = pd.DataFrame([json_data])
    try:
        # 'y' will be None, so we just ignore it with '_'
        X_processed, _ = train.preprocess(X_raw)

    except KeyError as e:
        # This will catch errors if the input JSON is missing a required
        # raw column (like 'datetime' or 'season')
        return jsonify({'error': f'Missing expected input feature: {str(e)}'}), 400

    log_predictions = model.predict(X_processed)

    # Inverse transform (from log space)
    predictions = np.expm1(log_predictions)
    predictions[predictions < 0] = 0

    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9696)


