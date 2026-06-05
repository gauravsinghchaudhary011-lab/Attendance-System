from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import os
import json
import sqlite3
from datetime import datetime
import face_recognition as fr
import pandas as pd
import cv2
import numpy as np
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_JSON = os.path.join(SCRIPT_DIR, 'students.json')
DB_PATH = os.path.join(SCRIPT_DIR, 'attendance.db')
ATTENDANCE_FOLDER = os.path.join(SCRIPT_DIR, 'attendance')

def load_students():
    if os.path.exists(STUDENTS_JSON):
        with open(STUDENTS_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_students(students):
    with open(STUDENTS_JSON, 'w') as f:
        json.dump(students, f, indent=4)

def get_attendance_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, user_id, timestamp FROM attendance ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()
    # Parse records to match expected format: id, name, date, time, confidence
    parsed_records = []
    students = load_students()
    for record in records:
        user_id = str(record[1])
        timestamp = record[2]
        # Find name by student_id
        name = "Unknown"
        for student_name, info in students.items():
            if info['student_id'] == user_id:
                name = student_name
                break
        # Parse timestamp (assuming format YYYY-MM-DD HH:MM:SS)
        try:
            date_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            date_str = date_time.strftime('%Y-%m-%d')
            time_str = date_time.strftime('%H:%M:%S')
        except ValueError:
            date_str = timestamp
            time_str = ""
        # Confidence not available in this table, set to None
        parsed_records.append((record[0], name, date_str, time_str, None))
    return parsed_records

@app.route('/')
def index():
    students = load_students()
    attendance_records = get_attendance_records()
    return render_template('index.html', students=students, attendance=attendance_records[:10])  # Show last 10 records

@app.route('/students')
def students():
    students = load_students()
    return render_template('students.html', students=students)

@app.route('/add_student', methods=['POST'])
def add_student():
    name = request.form['name'].strip()
    student_id = request.form['student_id'].strip()
    if name and student_id:
        students = load_students()
        students[name.lower()] = {'student_id': student_id}
        save_students(students)
        flash('Student added successfully!', 'success')
    else:
        flash('Name and Student ID are required.', 'error')
    return redirect(url_for('students'))

@app.route('/attendance')
def attendance():
    records = get_attendance_records()
    return render_template('attendance.html', records=records)

@app.route('/export_attendance')
def export_attendance():
    records = get_attendance_records()
    df = pd.DataFrame(records, columns=['ID', 'Name', 'Date', 'Time', 'Confidence'])
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = os.path.join(ATTENDANCE_FOLDER, filename)
    os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)
    df.to_csv(filepath, index=False)
    flash(f'Attendance exported to {filename}', 'success')
    return redirect(url_for('attendance'))

@app.route('/train_model')
def train_model():
    try:
        recognizer, label_map = fr.train_and_save_model()
        if recognizer:
            flash('Model trained successfully!', 'success')
        else:
            flash('No faces found for training.', 'error')
    except Exception as e:
        flash(f'Error training model: {str(e)}', 'error')
    return redirect(url_for('index'))

@app.route('/start_recognition')
def start_recognition():
    # This would ideally start the camera in a separate thread or process
    # For now, just flash a message
    flash('Recognition started. Check console for output.', 'info')
    # In a real implementation, you might use threading or subprocess to run run_attendance.py
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
