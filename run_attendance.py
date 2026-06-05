import cv2
import numpy as np
import os
import json
from datetime import datetime
import csv

# Import functions from the face_recognition module
from face_recognition import train_and_save_model, load_model_and_labels

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_FOLDER = os.path.join(SCRIPT_DIR, 'faces')
MODEL_FOLDER = os.path.join(SCRIPT_DIR, 'model')
ATTENDANCE_FOLDER = os.path.join(SCRIPT_DIR, 'attendance')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'face_model.yml')
LABELS_PATH = os.path.join(MODEL_FOLDER, 'label_map.json')
RECOGNITION_IMG_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 80

# --- Helper Functions ---

def ensure_folders():
    os.makedirs(FACES_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

STUDENTS_PATH = os.path.join(SCRIPT_DIR, 'students.json')
def load_student_data():
    if os.path.exists(STUDENTS_PATH):
        with open(STUDENTS_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_student_data(data):
    with open(STUDENTS_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def mark_attendance(student_name, student_id):
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')
    attendance_file = os.path.join(ATTENDANCE_FOLDER, f'attendance_{date_str}.csv')

    # Check if file exists to write header
    file_exists = os.path.isfile(attendance_file)
    
    # Check if already marked
    if file_exists:
        with open(attendance_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[1] == student_name:
                    return # Already marked

    # Mark attendance
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Name', 'StudentID'])
        writer.writerow([time_str, student_name, student_id])
    print(f"Attendance marked for {student_name}")

# --- Core Functions ---

def register_new_user():
    ensure_folders()
    name_input = input("Enter student's name: ")
    student_id = input(f"Enter student's student ID: ")

    if not name_input or not student_id:
        print("Name and Student ID cannot be empty.")
        return
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("\nPlease look at the camera. A picture will be taken for registration.")
    print("Press 'c' to capture the image.")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not open webcam.")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Registration - Press 'c' to capture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                # Normalize the name for the filename
                img_path = os.path.join(FACES_FOLDER, f"{name_input.lower()}.jpg")
                cv2.imwrite(img_path, face_roi)
                print(f"Face captured and saved to {img_path}")
                break
            elif len(faces) > 1:
                print("Multiple faces detected. Please ensure only one person is in the frame.")
            else:
                print("No face detected. Please position yourself in front of the camera.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save student data
    student_data = load_student_data()
    student_data[name_input.lower()] = {"student_id": student_id}
    save_student_data(student_data)
    print(f"{name_input} has been registered. Retraining model...")
    # Call the centralized training function
    train_and_save_model()
    print("Model training complete.")


def run_attendance_system():
    ensure_folders()
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please register users and train the model first.")
        return

    try:
        recognizer, idx_to_label = load_model_and_labels()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading model or labels: {e}")
        return

    student_data = load_student_data()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("\nStarting attendance system. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(cv2.resize(face_roi, RECOGNITION_IMG_SIZE))

            if confidence < CONFIDENCE_THRESHOLD:
                name = idx_to_label.get(label, "Unknown")
                student_id = student_data.get(name.lower(), {}).get('student_id', 'N/A')
                color = (0, 255, 0)
                mark_attendance(name, student_id)
                display_text = f"{name} ({student_id})"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                display_text = name

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Attendance System - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main Menu ---

def main():
    ensure_folders()  # Ensure all necessary folders exist at startup

    while True:
        print("\n--- Online Attendance System ---")
        print("1. Register New User")
        print("2. Train Model")
        print("3. Run Attendance System")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            register_new_user()
        elif choice == '2':
            print("\nTraining model...")
            train_and_save_model()
            print("Model training complete.")
        elif choice == '3':
            run_attendance_system()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
