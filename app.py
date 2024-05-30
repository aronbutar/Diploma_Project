import numpy as np
import sqlite3
import json
import logging
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import csv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='diagnosis.log', filemode='w')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

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
        self.symptoms = ["Fever (°C)", "Headache (Severity 0-5)", "Stomach pain (Severity 0-5)", "Cough (Severity 0-5)", "Chest pain (Severity 0-5)"]
        self.diagnoses = ["Viral Fever", "Malaria", "Typhoid", "Stomach problem", "Heart problem"]
        self.init_db()

        self.R_membership = np.random.rand(len(self.symptoms), len(self.diagnoses))
        self.R_non_membership = np.random.rand(len(self.symptoms), len(self.diagnoses))

        logging.info("MedicalDiagnosis initialized.")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
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
    return render_template('index.html', patients=patients, symptoms=md.symptoms, diagnoses=md.diagnoses)

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
            flash('Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    name = request.form['name']
    symptoms = []
    for symptom in md.symptoms:
        value = float(request.form[symptom])
        if "Fever" in symptom:
            normalized_value = md.normalize(value, 36, 41)
        else:
            normalized_value = md.normalize(value, 0, 5)
        symptoms.append(normalized_value)
    md.add_patient(name, symptoms)
    return redirect(url_for('index'))

@app.route('/diagnose')
@login_required
def diagnose():
    patients_data = md.get_patients()
    Q_membership = []
    Q_non_membership = []
    patient_names = []

    for patient in patients_data:
        name, symptoms = patient[1], json.loads(patient[2])
        patient_names.append(name)
        Q_membership.append(symptoms)
        Q_non_membership.append([1-s for s in symptoms])

    Q_membership = np.array(Q_membership)
    Q_non_membership = np.array(Q_non_membership)

    iterations = 3
    for iteration in range(iterations):
        T_membership, T_non_membership = md.max_min_max_composition((Q_membership, Q_non_membership), (md.R_membership, md.R_non_membership))
        SR = md.calculate_SR((T_membership, T_non_membership))

    diagnoses = []
    for i, patient in enumerate(patient_names):
        patient_diagnoses = []
        for j, diagnosis in enumerate(md.diagnoses):
            patient_diagnoses.append({
                'diagnosis': diagnosis,
                'membership': T_membership[i, j],
                'non_membership': T_non_membership[i, j],
                'SR': SR[i, j]
            })
        diagnoses.append({'patient': patient, 'diagnoses': patient_diagnoses})

    return render_template('diagnosis.html', diagnoses=diagnoses)

@app.route('/export')
@login_required
def export_data():
    patients = md.get_patients()
    with open('patients.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'name', 'symptoms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for patient in patients:
            writer.writerow({'id': patient[0], 'name': patient[1], 'symptoms': patient[2]})

    return send_file('patients.csv', as_attachment=True)

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
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
