from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Setup logging
logging.basicConfig(level=logging.DEBUG)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

db_path = 'patients.db'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

def create_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        password TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        symptoms TEXT NOT NULL,
                        diagnosis TEXT)''')
    conn.commit()
    conn.close()

create_tables()

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class MedicalDiagnosis:
    def __init__(self):
        self.symptoms = ['temperature', 'headache', 'cough', 'fatigue', 'sore_throat']
        self.diagnostics = ['Healthy', 'Common Cold', 'Flu', 'Allergy', 'Bronchitis', 'Pneumonia']
        self.setup_fuzzy_system()

    def setup_fuzzy_system(self):
        # Antecedents
        self.temperature = ctrl.Antecedent(np.arange(35, 41, 0.1), 'temperature')
        self.headache = ctrl.Antecedent(np.arange(0, 11, 1), 'headache')
        self.cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')
        self.fatigue = ctrl.Antecedent(np.arange(0, 11, 1), 'fatigue')
        self.sore_throat = ctrl.Antecedent(np.arange(0, 11, 1), 'sore_throat')

        # Consequent
        self.diagnosis = ctrl.Consequent(np.arange(0, 101, 1), 'diagnosis')

        # Membership functions
        self.temperature.automf(names=['low', 'medium', 'high', 'very_high'])
        self.headache.automf(names=['low', 'medium', 'high', 'very_high'])
        self.cough.automf(names=['low', 'medium', 'high', 'very_high'])
        self.fatigue.automf(names=['low', 'medium', 'high', 'very_high'])
        self.sore_throat.automf(names=['low', 'medium', 'high', 'very_high'])

        # Diagnosis membership functions
        self.diagnosis['healthy'] = fuzz.trimf(self.diagnosis.universe, [0, 0, 20])
        self.diagnosis['common_cold'] = fuzz.trimf(self.diagnosis.universe, [20, 30, 40])
        self.diagnosis['flu'] = fuzz.trimf(self.diagnosis.universe, [40, 50, 60])
        self.diagnosis['allergy'] = fuzz.trimf(self.diagnosis.universe, [60, 70, 80])
        self.diagnosis['bronchitis'] = fuzz.trimf(self.diagnosis.universe, [80, 90, 100])
        self.diagnosis['pneumonia'] = fuzz.trimf(self.diagnosis.universe, [100, 110, 120])

        # Rules
        self.rules = [
            # Healthy
            ctrl.Rule(self.temperature['low'] & self.cough['low'] & self.fatigue['low'], self.diagnosis['healthy']),
            
            # Common Cold
            ctrl.Rule(self.temperature['medium'] & self.headache['low'] & self.sore_throat['medium'], self.diagnosis['common_cold']),
            ctrl.Rule(self.temperature['medium'] & self.cough['medium'] & self.sore_throat['medium'], self.diagnosis['common_cold']),
            ctrl.Rule(self.temperature['medium'] & self.cough['medium'] & self.fatigue['medium'], self.diagnosis['common_cold']),
            
            # Flu
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.headache['medium'], self.diagnosis['flu']),
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.fatigue['medium'], self.diagnosis['flu']),
            ctrl.Rule(self.temperature['medium'] & self.fatigue['high'] & self.sore_throat['high'], self.diagnosis['flu']),
            
            # Allergy
            ctrl.Rule(self.temperature['low'] & self.cough['low'] & self.sore_throat['medium'], self.diagnosis['allergy']),
            ctrl.Rule(self.temperature['medium'] & self.headache['medium'] & self.sore_throat['high'], self.diagnosis['allergy']),
            ctrl.Rule(self.temperature['high'] & self.headache['low'] & self.cough['low'], self.diagnosis['allergy']),
            
            # Bronchitis
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.fatigue['medium'], self.diagnosis['bronchitis']),
            ctrl.Rule(self.temperature['medium'] & self.cough['high'] & self.headache['high'], self.diagnosis['bronchitis']),
            ctrl.Rule(self.temperature['low'] & self.headache['high'] & self.fatigue['high'], self.diagnosis['bronchitis']),
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.headache['medium'], self.diagnosis['bronchitis']),
            ctrl.Rule(self.temperature['low'] & self.cough['high'] & self.sore_throat['high'], self.diagnosis['bronchitis']),
            ctrl.Rule(self.temperature['medium'] & self.cough['medium'] & self.fatigue['high'], self.diagnosis['bronchitis']),
            
            # Pneumonia
            ctrl.Rule(self.temperature['very_high'] & self.cough['very_high'] & self.headache['very_high'] & self.fatigue['very_high'] & self.sore_throat['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['very_high'] & self.cough['very_high'] & self.fatigue['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.fatigue['high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['high'] & self.headache['high'] & self.cough['medium'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['high'] & self.cough['very_high'] & self.sore_throat['very_high'], self.diagnosis['pneumonia']),
            
            # Emphasize severe symptoms
            ctrl.Rule(self.temperature['very_high'] & self.cough['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['very_high'] & self.headache['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['very_high'] & self.fatigue['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['very_high'] & self.sore_throat['very_high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['high'] & self.cough['high'] & self.headache['high'] & self.fatigue['high'] & self.sore_throat['high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['high'] & self.cough['very_high'] & self.headache['high'], self.diagnosis['pneumonia']),
            ctrl.Rule(self.temperature['very_high'] & self.headache['very_high'] & self.fatigue['high'], self.diagnosis['pneumonia']),
        ]

        self.diagnosis_ctrl = ctrl.ControlSystem(self.rules)
        self.diagnosis_sim = ctrl.ControlSystemSimulation(self.diagnosis_ctrl)

    def diagnose(self, patient):
        symptoms = patient[2].split(',')
        if len(symptoms) != len(self.symptoms):
            raise ValueError("Incorrect number of symptoms provided")

        symptoms_dict = {self.symptoms[i]: float(symptoms[i]) for i in range(len(self.symptoms))}
        for symptom, value in symptoms_dict.items():
            self.diagnosis_sim.input[symptom] = value

        self.diagnosis_sim.compute()
        diagnosis_value = self.diagnosis_sim.output['diagnosis']
        diagnosis_label = self.get_diagnosis_label(diagnosis_value)

        return diagnosis_label

    def get_diagnosis_label(self, diagnosis_value):
        if diagnosis_value < 20:
            return "Healthy"
        elif diagnosis_value < 40:
            return "Common Cold"
        elif diagnosis_value < 60:
            return "Flu"
        elif diagnosis_value < 80:
            return "Allergy"
        elif diagnosis_value < 100:
            return "Bronchitis"
        else:
            return "Pneumonia"
        
class IntuitionisticFuzzyDiagnosis:
    def __init__(self):
        self.symptoms = ['temperature', 'headache', 'cough', 'fatigue', 'sore_throat']
        self.diagnoses = ['Healthy', 'Common Cold', 'Flu', 'Allergy', 'Bronchitis', 'Pneumonia']
        
        # Revised R Membership matrix
        self.R_membership = np.array([
            [0.4, 0.6, 0.8, 0.4, 0.6, 0.9],  # Temperature
            [0.1, 0.6, 0.7, 0.3, 0.7, 0.8],  # Headache
            [0.1, 0.7, 0.8, 0.6, 0.8, 0.9],  # Cough
            [0.2, 0.5, 0.7, 0.3, 0.7, 0.8],  # Fatigue
            [0.1, 0.7, 0.6, 0.5, 0.7, 0.8]   # Sore Throat
        ])
        
        # Non-membership matrix, complement of the membership matrix
        self.R_non_membership = 1 - self.R_membership

        # Weights for each symptom relative to each diagnosis
        self.weights = np.array([
            [0.5, 0.7, 0.9, 0.6, 0.8, 1.0],  # Temperature Weight
            [0.4, 0.6, 0.7, 0.5, 0.7, 0.8],  # Headache Weight
            [0.4, 0.8, 1.0, 0.6, 0.9, 1.0],  # Cough Weight
            [0.3, 0.5, 0.8, 0.4, 0.7, 0.9],  # Fatigue Weight
            [0.3, 0.7, 0.7, 0.6, 0.7, 0.8]   # Sore Throat Weight
        ])

    def max_min_max_composition(self, Q_membership, Q_non_membership):
        T_membership = np.zeros((Q_membership.shape[0], self.R_membership.shape[1]))
        T_non_membership = np.zeros((Q_membership.shape[0], self.R_membership.shape[1]))
        
        # Apply weights to the R_membership matrix before composition
        weighted_R_membership = self.R_membership * self.weights
        logging.debug("Weighted R Membership Matrix: %s", weighted_R_membership)

        for i in range(Q_membership.shape[0]):
            for j in range(weighted_R_membership.shape[1]):
                T_membership[i, j] = np.max(np.minimum(Q_membership[i], weighted_R_membership[:, j]))
                T_non_membership[i, j] = np.min(np.maximum(Q_non_membership[i], self.R_non_membership[:, j]))
        
        logging.debug("T Membership Matrix: %s", T_membership)
        logging.debug("T Non-Membership Matrix: %s", T_non_membership)

        return T_membership, T_non_membership

    def calculate_SR(self, T_membership, T_non_membership):
        pi_T = 1 - (T_membership + T_non_membership)
        pi_T[pi_T < 0] = 0

        SR = T_membership - T_non_membership + pi_T * (1 - np.abs(T_membership - T_non_membership))
        
        logging.debug("Pi_T: %s", pi_T)
        logging.debug("SR Values: %s", SR)
        
        return SR

    def create_q_matrix(self, patients):
        q_matrix_membership = []
        q_matrix_non_membership = []
        for patient in patients:
            membership_row = []
            non_membership_row = []
            if isinstance(patient['symptoms'], list):
                patient_symptoms_dict = dict(zip(self.symptoms, patient['symptoms']))
            else:
                patient_symptoms_dict = patient['symptoms']

            for symptom, value in patient_symptoms_dict.items():
                memberships = self.linear_membership(value, symptom)
                membership_row.append(memberships[0])
                non_membership_row.append(memberships[1])
            q_matrix_membership.append(membership_row)
            q_matrix_non_membership.append(non_membership_row)
        
        logging.debug("Q Membership Matrix: %s", np.array(q_matrix_membership))
        logging.debug("Q Non-Membership Matrix: %s", np.array(q_matrix_non_membership))

        return np.array(q_matrix_membership), np.array(q_matrix_non_membership)

    def linear_membership(self, value, symptom):
        if symptom == 'temperature':
            if value < 36.5:
                membership = 0.4
            elif 36.5 <= value <= 37.5:
                membership = 0.8
            elif 37.5 <= value <= 38.5:
                membership = 0.9
            else:
                membership = 1.0
        else:
            membership = max(0.8, min(1, value / 10))
        non_membership = 1 - membership
        return (membership, non_membership)

    def diagnose(self, patient_symptoms):
        if isinstance(patient_symptoms, list):
            patient_symptoms_dict = dict(zip(self.symptoms, patient_symptoms))
        else:
            patient_symptoms_dict = patient_symptoms

        Q_membership, Q_non_membership = self.create_q_matrix([{'symptoms': patient_symptoms_dict}])
        Q_membership = Q_membership[0]
        Q_non_membership = Q_non_membership[0]

        logging.debug(f"Q_membership: {Q_membership}")
        logging.debug(f"Q_non_membership: {Q_non_membership}")

        T_membership, T_non_membership = self.max_min_max_composition(Q_membership, Q_non_membership)
        SR = self.calculate_SR(T_membership, T_non_membership)

        diagnosis_result = []
        for i in range(T_membership.shape[1]):
            diagnosis_result.append({
                'diagnosis': self.diagnoses[i],
                'membership': round(float(T_membership[0, i]), 2),
                'non_membership': round(float(T_non_membership[0, i]), 2),
                'SR': round(float(SR[0, i]), 2)
            })

        logging.debug(f"Diagnosis results: {diagnosis_result}")

        diagnosis_result.sort(key=lambda x: (x['SR'], self.diagnoses.index(x['diagnosis'])), reverse=True)

        return diagnosis_result


# Instantiate the diagnosis systems
md = MedicalDiagnosis()
ifd = IntuitionisticFuzzyDiagnosis()

@app.route('/')
@login_required
def index():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients')
    patients = cursor.fetchall()
    conn.close()
    return render_template('index.html', patients=patients, symptoms=md.symptoms, diagnostics=md.diagnostics)

@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    name = request.form['name']
    symptoms = [request.form.get(symptom, '') for symptom in md.symptoms]
    if '' in symptoms:
        flash('Please provide all symptom values', 'danger')
        return redirect(url_for('index'))

    try:
        symptoms = list(map(float, symptoms))
    except ValueError:
        flash('Please provide valid numerical values for symptoms', 'danger')
        return redirect(url_for('index'))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO patients (name, symptoms) VALUES (?, ?)', (name, ','.join(map(str, symptoms))))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/remove_patient/<int:patient_id>')
@login_required
def remove_patient(patient_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM patients WHERE id=?', (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/diagnose_patient/<int:patient_id>')
@login_required
def diagnose_patient(patient_id):
    logging.debug("Diagnosing patient with ID: %s", patient_id)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id=?', (patient_id,))
    patient = cursor.fetchone()
    conn.close()

    if patient:
        try:
            diagnosis_result = md.diagnose(patient)
            logging.debug("Fuzzy diagnosis result: %s", diagnosis_result)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

        patient_symptoms = [float(value) for value in patient[2].split(',')]
        symptoms_values = dict(zip(md.symptoms, patient_symptoms))
        try:
            ifd_diagnosis_results = ifd.diagnose(patient_symptoms)
            logging.debug("Intuitionistic fuzzy diagnosis results: %s", ifd_diagnosis_results)
            if not ifd_diagnosis_results:
                flash('No diagnosis results available for Intuitionistic Fuzzy Logic', 'danger')
                return redirect(url_for('index'))
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

        return render_template('diagnosis_result.html', patient_id=patient_id, patient=patient[1], diagnosis_result=diagnosis_result, ifd_diagnosis_results=ifd_diagnosis_results, symptoms=md.symptoms, symptoms_values=symptoms_values)
    else:
        flash('Patient not found', 'danger')
        return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if (request.method == 'POST'):
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1], user[2])
            login_user(user_obj)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if (request.method == 'POST'):
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)