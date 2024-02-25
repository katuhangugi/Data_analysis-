import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_file='hypertension_app.db'):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                username TEXT PRIMARY KEY,
                email TEXT,
                phone TEXT,
                password TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                username TEXT PRIMARY KEY,
                password TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_records (
                patient_username TEXT,
                record_text TEXT,
                recorded_at DATETIME,
                FOREIGN KEY (patient_username) REFERENCES patients(username)
            )
        ''')

        # Commit changes

    def commit_changes(self):
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

    def add_patient(self, username, email, phone, password):
        self.cursor.execute("INSERT INTO patients VALUES (?, ?, ?, ?)", (username, email, phone, password))
        self.commit_changes()

    def add_doctor(self, username, password):
        self.cursor.execute("INSERT INTO doctors VALUES (?, ?)", (username, password))
        self.commit_changes()

    def get_patient_data(self, patient_id):
        self.cursor.execute("SELECT * FROM health_records WHERE patient_username=?", (patient_id,))
        records = self.cursor.fetchall()
        return {'records': records}

    def validate_patient(self, username, password):
        self.cursor.execute("SELECT * FROM patients WHERE username=? AND password=?", (username, password))
        return self.cursor.fetchone() is not None

    def validate_doctor(self, username, password):
        self.cursor.execute("SELECT * FROM doctors WHERE username=? AND password=?", (username, password))
        return self.cursor.fetchone() is not None

    def get_all_patients(self):
        self.cursor.execute("SELECT * FROM patients")
        patients = self.cursor.fetchall()
        return [{'username': patient[0], 'email': patient[1], 'phone': patient[2]} for patient in patients]

    def add_health_record(self, patient_username, record_text):
        recorded_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute("INSERT INTO health_records VALUES (?, ?, ?)", (patient_username, record_text, recorded_at))
        self.commit_changes()
