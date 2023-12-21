from flask import Flask, render_template, request

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and scaler
model = LogisticRegression(max_iter=1000)
scaler = StandardScaler()

# Load the dataset (replace with the actual path to your dataset)
data = pd.read_csv("C:/Users/PC/Diabetes/dataset.csv")

# Identify independent variables (features) and the target variable (Outcome)
X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]  # Target variable

# Standardize the features
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)


# Train the model
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        values = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Standardize input features
        values_scaled = scaler.transform([values])

        # Debug prints for form data and prediction
        print("Form Data:", values)
        print("Scaled Input:", values_scaled)

        # Make prediction
        prediction = model.predict(values_scaled)

        # Debug print for prediction result
        print("Prediction:", prediction)

        return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)


print("Form Data:", request.form)