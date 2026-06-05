import cv2
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FOLDER = os.path.join(SCRIPT_DIR, 'faces')
MODEL_FOLDER = os.path.join(SCRIPT_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'face_model.yml')
LABELS_PATH = os.path.join(MODEL_FOLDER, 'label_map.json')
RECOGNITION_IMG_SIZE = (200, 200)

def train_and_save_model():
    if not os.path.isdir(REFERENCE_FOLDER) or not os.listdir(REFERENCE_FOLDER):
        print(f"Reference folder '{REFERENCE_FOLDER}' is empty or doesn't exist.")
        return None, None

    faces_data = []
    labels = []
    label_map = {}
    label_count = 0

    for filename in os.listdir(REFERENCE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(REFERENCE_FOLDER, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Normalize the label to lowercase
            person_label = os.path.splitext(filename)[0].lower()
            if person_label not in label_map:
                label_map[person_label] = label_count
                label_count += 1
            
            faces_data.append(cv2.resize(img, RECOGNITION_IMG_SIZE))
            labels.append(label_map[person_label])

    if not faces_data:
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_data, np.array(labels, dtype=np.int32))
    
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    recognizer.save(MODEL_PATH)

    idx_to_label = {v: k for k, v in label_map.items()}
    with open(LABELS_PATH, 'w') as f:
        json.dump(idx_to_label, f)

    return recognizer, idx_to_label

def load_model_and_labels():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, 'r') as f:
        idx_to_label_str_keys = json.load(f)
        idx_to_label = {int(k): v for k, v in idx_to_label_str_keys.items()}

    return recognizer, idx_to_label
