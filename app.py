from flask import Flask, request, render_template, send_file
import pickle
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load the model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/input_details', methods=['POST'])
def input_details():
    patient_name = request.form['patient_name']
    contact_number = request.form['contact_number']
    return render_template('index.html', patient_name=patient_name, contact_number=contact_number)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract patient details
        patient_name = request.form['patient_name']
        contact_number = request.form['contact_number']

        # Extract prediction features
        feature_keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features = []
        for key in feature_keys:
            value = request.form.get(key)
            if value is None:
                raise ValueError(f"Missing value for {key}")
            features.append(float(value))

        final_features = [np.array(features)]

        # Predict using the model
        prediction = model.predict(final_features)
        output = prediction[0]

        # Return the result
        prediction_text = 'Heart Disease Prediction: {}'.format('Yes' if output == 1 else 'No')
        return render_template('index.html',
                               prediction_text=prediction_text,
                               patient_name=patient_name,
                               contact_number=contact_number,
                               prediction=output)
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e), patient_name=patient_name, contact_number=contact_number)


@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        # Extract patient details and prediction
        patient_name = request.form['patient_name']
        contact_number = request.form['contact_number']
        prediction = request.form['prediction']

        # Create a PDF report
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("Heart Disease Prediction Report")

        pdf.drawString(100, 750, "Heart Disease Prediction Report")
        pdf.drawString(100, 730, f"Patient Name: {patient_name}")
        pdf.drawString(100, 710, f"Contact Number: {contact_number}")
        pdf.drawString(100, 690, f"Prediction: {'Yes' if prediction == '1' else 'No'}")

        pdf.showPage()
        pdf.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='Heart_Disease_Prediction_Report.pdf', mimetype='application/pdf')

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
