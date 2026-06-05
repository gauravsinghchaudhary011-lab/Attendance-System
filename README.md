# Face Recognition Attendance System

This project implements a face recognition-based attendance system using OpenCV and machine learning algorithms, with a web-based frontend for easy management.

## Features

- Face detection and recognition using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm
- Real-time attendance marking via camera feed
- SQLite database for storing attendance records
- CSV export of attendance data
- Web-based frontend for student management, attendance viewing, and model training
- Dashboard with system status and recent attendance

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Flask
- Flask-WTF
- Pandas
- SQLite3 (built-in with Python)

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Place reference images in the `faces/` folder. Each image should be named after the person (e.g., `john.jpg`).

3. Run the web application:
   ```
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000`

## Usage

### Web Interface

- **Dashboard**: View system status, recent attendance, and quick actions.
- **Students**: Add and manage student information.
- **Attendance**: View all attendance records and export to CSV.
- **Train Model**: Train the face recognition model using images in the `faces/` folder.
- **Start Recognition**: Begin real-time attendance marking via camera (opens in separate window).

### CLI Usage (Alternative)

1. Run the training script to train the model:
   ```
   python face_recognition.py
   ```

2. Run the attendance system:
   ```
   python run_attendance.py
   ```

## File Structure

- `app.py`: Main Flask web application.
- `face_recognition.py`: Contains functions for training and loading the face recognition model.
- `run_attendance.py`: Main script for running the attendance system with camera feed.
- `students.json`: JSON file containing student information.
- `attendance.db`: SQLite database for storing attendance records.
- `faces/`: Folder containing reference face images.
- `model/`: Folder containing the trained model and label mappings.
- `attendance/`: Folder containing exported CSV files of attendance data.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS and JavaScript files for styling and interactivity.

## Notes

- The system uses grayscale images for face recognition.
- Confidence threshold can be adjusted in the `run_attendance.py` script.
- Attendance is marked only when a face is recognized with sufficient confidence.
- The web interface allows for easy management without command-line interaction.
