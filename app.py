from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('water_potability_model.pkl')  # Update this with your model's path
scaler = joblib.load('scaler.pkl')  # Ensure you save the scaler during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def show_graphs():
    # List of feature columns for dynamic rendering in the HTML
    feature_columns = ['Ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    return render_template('graphs.html', feature_columns=feature_columns)


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    input_values = [
        float(request.form['Ph']),
        float(request.form['Hardness']),
        float(request.form['Solids']),
        float(request.form['Chloramines']),
        float(request.form['Sulfate']),
        float(request.form['Conductivity']),
        float(request.form['Organic_carbon']),
        float(request.form['Trihalomethanes']),
        float(request.form['Turbidity'])
    ]

    # Scale the input data
    input_data = scaler.transform([input_values])

    # Prediction
    prediction = model.predict(input_data)

    # Convert numerical prediction to text
    result = "Potable" if prediction[0] == 1 else "Not Potable"

    # Render the result template with the text result
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
