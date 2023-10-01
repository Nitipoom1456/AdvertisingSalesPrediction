# advertising_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

# Step 1: Read the advertising data using Pandas
data = pd.read_csv("advertising.csv")

# Step 2: Split data into features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Step 6: Create a pipeline
pipeline = Pipeline([
    ('model', LinearRegression())
])

# Step 7: Save the pipeline (including the model) using joblib
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model.joblib')

# Step 8: Load the model
loaded_pipeline = joblib.load('model.joblib')

# Step 9: Streamlit web app
st.title('Advertising Sales Prediction')

st.sidebar.header('User Input Features')

# Create input sliders for TV, Radio, and Newspaper
tv = st.sidebar.slider('TV Advertising Budget', min_value=0, max_value=300, value=150)
radio = st.sidebar.slider('Radio Advertising Budget', min_value=0, max_value=50, value=25)
newspaper = st.sidebar.slider('Newspaper Advertising Budget', min_value=0, max_value=100, value=50)

# Create a feature vector for prediction
input_features = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})

# Add a "Predict" button
predict_button = st.sidebar.button("Predict")

# Make a prediction only when the button is clicked
if predict_button:
    # Make a prediction using the loaded pipeline (including the model)
    predicted_sales = loaded_pipeline.predict(input_features)
    st.write('Predicted Sales:', predicted_sales[0])
