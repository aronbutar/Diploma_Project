import os
import numpy as np
import sqlite3
import json
import logging
from flask import Flask, request, render_template, redirect, url_for, send_file, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import csv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='diagnosis.log', filemode='w')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Example users (in-memory, for demonstration)
users = {
    'admin': User(id=1, username='admin', password='secret')
}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class MedicalDiagnosis:
    def __init__(self, db_path='medical_diagnosis.db'):
        self.db_path = db_path
        self.symptoms = ['Fever', 'Headache', 'Cough', 'Fatigue', 'Sore Throat', 'Shortness of Breath', 'Nausea']
        self.diagnostics = ['Unknown', 'Respiratory_allergy', 'Common_cold', 'Mild_viral_infection', 'Sinusitis', 'Flu', 'Bronchitis', 'Bacterial_infection', 'Pneumonia']
        self.init_db()

        self.R_membership = np.random.rand(len(self.symptoms), len(self.diagnostics))
        self.R_non_membership = np.random.rand(len(self.symptoms), len(self.diagnostics))

        logging.info("MedicalDiagnosis initialized.")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS patients')  # Drop the table if it exists
        cursor.execute('''
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY,
                name TEXT,
                symptoms TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_patient(self, name, symptoms):
        normalized_symptoms = json.dumps(symptoms)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO patients (name, symptoms) VALUES (?, ?)', (name, normalized_symptoms))
        conn.commit()
        conn.close()
        logging.info(f"Added patient: {name} with symptoms {symptoms}")
    
    def get_patients(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients')
        rows = cursor.fetchall()
        conn.close()
        logging.info(f"Fetched patients data: {rows}")
        return rows

    def normalize(self, value, min_value, max_value):
        """
        Normalize a real-world value to the range [0, 1].
        """
        return (value - min_value) / (max_value - min_value)
    
    def max_min_max_composition(self, A, R):
        B_membership = np.max(np.minimum(A[0][:, None], R[0]), axis=0)
        B_non_membership = np.min(np.maximum(A[1][:, None], R[1]), axis=0)
        return B_membership, B_non_membership

    def calculate_SR(self, R):
        membership_diff = 1 - (R[0] + R[1])
        non_membership_diff = 1 - membership_diff
        return membership_diff - non_membership_diff

    def refine_R(self, Q, T, R):
        refined_R_membership = np.max(np.minimum(Q[0][:, None], T[0]), axis=0)
        refined_R_non_membership = np.min(np.maximum(Q[1][:, None], T[1]), axis=0)
        return refined_R_membership, refined_R_non_membership

md = MedicalDiagnosis()

@app.route('/')
@login_required
def index():
    patients = md.get_patients()
    return render_template('index.html', patients=patients, symptoms=md.symptoms, diagnostics=md.diagnostics)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    name = request.form['name']
    symptoms = []
    for symptom in md.symptoms:
        try:
            value = float(request.form[symptom])
            if value < 0 or value > 5:
                flash(f"Invalid value for {symptom}. Please enter a value between 0 and 5.", 'danger')
                return redirect(url_for('index'))
            normalized_value = md.normalize(value, 0, 5)  # Adjust normalization as needed for symptoms
            symptoms.append(normalized_value)
        except ValueError:
            flash(f"Invalid input for {symptom}. Please enter a numeric value.", 'danger')
            return redirect(url_for('index'))
    md.add_patient(name, symptoms)
    flash('Patient added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/get_patients', methods=['GET'])
@login_required
def get_patients():
    patients = md.get_patients()
    patient_list = [{'id': patient[0], 'name': patient[1]} for patient in patients]
    return jsonify(patient_list)

@app.route('/diagnose_patient/<int:patient_id>')
@login_required
def diagnose_patient(patient_id):
    try:
        patients_data = md.get_patients()
        patient_data = next((p for p in patients_data if p[0] == patient_id), None)
        if not patient_data:
            flash('Patient not found.', 'warning')
            return redirect(url_for('index'))
        
        name, symptoms = patient_data[1], json.loads(patient_data[2])
        Q_membership = np.array([symptoms])
        Q_non_membership = np.array([[1-s for s in symptoms]])

        logging.info(f"Q_membership: {Q_membership}")
        logging.info(f"Q_non_membership: {Q_non_membership}")

        if not Q_membership.size or not Q_non_membership.size:
            flash('No valid symptoms data found.', 'warning')
            return redirect(url_for('index'))

        iterations = 3
        for iteration in range(iterations):
            logging.info(f"Iteration {iteration+1}")
            T_membership, T_non_membership = md.max_min_max_composition((Q_membership, Q_non_membership), (md.R_membership, md.R_non_membership))
            SR = md.calculate_SR((T_membership, T_non_membership))

        diagnoses = []
        for j, diagnosis in enumerate(md.diagnostics):
            diagnoses.append({
                'diagnosis': diagnosis,
                'membership': T_membership[0, j],
                'non_membership': T_non_membership[0, j],
                'SR': SR[0, j]
            })

        logging.info(f"Diagnoses for patient {name}: {diagnoses}")
        return render_template('diagnosis.html', diagnoses=[{'patient': name, 'diagnoses': diagnoses}])
    except Exception as e:
        logging.error(f"Error during diagnosis: {e}")
        flash('An error occurred during diagnosis.', 'danger')
        return redirect(url_for('index'))

@app.route('/export')
@login_required
def export_data():
    patients = md.get_patients()
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], 'patients.csv')
    logging.info(f"Export path: {export_path}")
    
    try:
        with open(export_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'name', 'symptoms']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for patient in patients:
                writer.writerow({'id': patient[0], 'name': patient[1], 'symptoms': patient[2]})

        logging.info(f"Exported data to {export_path}")
    except Exception as e:
        logging.error(f"Failed to write CSV file: {e}")
        flash('Failed to export data.', 'danger')
        return redirect(url_for('index'))

    return send_file(export_path, as_attachment=True)

@app.route('/import', methods=['POST'])
@login_required
def import_data():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        reader = csv.DictReader(file.stream)
        for row in reader:
            name = row['name']
            symptoms = json.loads(row['symptoms'])
            md.add_patient(name, symptoms)
    flash('Data imported successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
