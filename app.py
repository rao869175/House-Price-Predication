import os
os.system("pip install scikit-learn")
import os
os.system("pip install scikit-learn")
import gradio as gr
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('diseases4.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the feature names directly (replace with your actual feature names from your dataset)
feature_names = [
    "Size (sq ft)",  # Replace with your actual feature names
    "Number of Bedrooms",
    "Number of Bathrooms",
    "Age of House",
    "Distance to City Center"
]

# Placeholder data for fitting the scaler (ideally, use the original training data)
# Example training data for fitting the scaler (replace with actual training data or load from a file)
dummy_training_data = np.array([
    [1500, 3, 2, 10, 5],
    [2000, 4, 3, 15, 10],
    [2500, 5, 4, 20, 15],
    [1800, 3, 2, 12, 8],
    [2200, 4, 3, 8, 12]
])

# StandardScaler - fit on the dummy training data
scaler = StandardScaler()
scaler.fit(dummy_training_data)  # Fit the scaler on the training data

# Define prediction function
def predict_house_price(*features):
    input_data = np.array(features).reshape(1, -1)  # Convert input to NumPy array
    input_scaled = scaler.transform(input_data)  # Scale input using the fitted scaler
    prediction = model.predict(input_scaled)  # Make prediction
    return f"🏠 Predicted House Price: ${round(prediction[0], 2)}"

# Create input fields dynamically based on the feature names
inputs = [
    gr.Number(label="Size (sq ft)", value=2000),
    gr.Number(label="Number of Bedrooms", value=3),
    gr.Number(label="Number of Bathrooms", value=2),
    gr.Number(label="Age of House", value=10),
    gr.Number(label="Distance to City Center", value=5)
]

# Define Gradio Interface
demo = gr.Interface(
    fn=predict_house_price,
    inputs=inputs,
    outputs="text",
    title="House Price Prediction",
    description="Enter the house features below to predict its price."
)

# Launch Gradio app
demo.launch()
