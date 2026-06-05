# TODO List for Adding Frontend to Face Recognition Attendance System

## Step 1: Update requirements.txt
- Add Flask and related dependencies (e.g., flask-wtf for forms).

## Step 2: Create Flask app (app.py)
- Set up main Flask application with routes for dashboard, student management, attendance viewing, and model training.
- Integrate students.json and attendance.db.

## Step 3: Create templates folder and HTML files
- Create templates/base.html (base template).
- Create templates/index.html (dashboard).
- Create templates/students.html (student management).
- Create templates/attendance.html (attendance viewing).
- Create templates/train.html (model training interface).

## Step 4: Create static folder and assets
- Create static/css/style.css (custom styles, possibly with Bootstrap).
- Create static/js/script.js (any necessary JavaScript).

## Step 5: Modify run_attendance.py if needed
- Add API endpoint or integration for triggering camera recognition from web.

## Step 6: Update README.md
- Add instructions for running the frontend, including new dependencies and usage.

## Step 7: Install dependencies and test
- Install new packages.
- Run Flask app locally and test functionality.

## Step 8: Final testing and adjustments
- Ensure ML training works via web, attendance viewing, student management.
- Handle camera integration if possible.
