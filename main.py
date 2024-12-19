import cv2
import face_recognition
import os
import numpy as np
from sklearn.svm import SVC
import joblib

# Path dataset dan model
dataset_path = r"dataset/dataset/"
model_path = "svm_face_model.pkl"

# Validasi path dataset
if not os.path.exists(dataset_path):
    print(f"Error: Path dataset '{dataset_path}' tidak ditemukan. Periksa kembali!")
    exit()

def load_face_encodings(dataset_path):
    face_encodings = []
    face_names = []

    # Iterasi folder orang
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(image)

                    if face_encoding:
                        face_encodings.append(face_encoding[0])
                        face_names.append(person_name)
                except Exception as e:
                    print(f"Error pada file {img_path}: {e}")

    return face_encodings, face_names

def train_svm_model(encodings, names):
    print("Melatih model SVM...")
    model = SVC(kernel="linear", probability=True)
    model.fit(encodings, names)
    joblib.dump(model, model_path)
    print("Model berhasil dilatih dan disimpan!")

def recognize_faces_svm(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"

        probabilities = model.predict_proba([face_encoding])[0]
        best_match_index = np.argmax(probabilities)
        if probabilities[best_match_index] > 0.6:
            name = model.classes_[best_match_index]

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

if not os.path.exists(model_path):
    print("Model tidak ditemukan. Memuat dataset wajah dan melatih model...")
    known_encodings, known_names = load_face_encodings(dataset_path)
    if known_encodings and known_names:
        train_svm_model(known_encodings, known_names)
    else:
        print("Dataset kosong atau tidak valid!")
        exit()
else:
    print("Memuat model SVM yang sudah ada...")
    model = joblib.load(model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error membuka kamera!")
        break

    recognize_faces_svm(frame, model)
    cv2.imshow("Face Recognition with SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
