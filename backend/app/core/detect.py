import os
import cv2
import pickle
import numpy as np
import random
from datetime import datetime

class FaceDetectionRecognition:
    def __init__(self, embeddings_path, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.fd_model_path = os.path.join(self.model_dir, "face_detection_yunet.onnx")
        self.fr_model_path = os.path.join(self.model_dir, "face_recognition_sface.onnx")

        self.check_models()
        self.conf_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.recognition_threshold = 0.5

        self.init_face_detector()
        self.init_face_recognizer()
        self.load_embeddings(embeddings_path)

    def generate_uid_face(self):
        return random.randint(10000000, 99999999)

    def check_models(self):
        if not os.path.exists(self.fd_model_path) or not os.path.exists(self.fr_model_path):
            raise FileNotFoundError("Face detection/recognition model not found in {}".format(self.model_dir))

    def init_face_detector(self):
        self.face_detector = cv2.FaceDetectorYN.create(
            model=self.fd_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def init_face_recognizer(self):
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            model=self.fr_model_path,
            config=""
        )

    def load_embeddings(self, embeddings_path):
        self.embeddings_db = {}
        if not os.path.exists(embeddings_path):
            return
        if os.path.getsize(embeddings_path) == 0:
            return
        try:
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            # Handle old/new format
            if data and isinstance(list(data.values())[0], np.ndarray):
                new_db = {}
                for name, embeddings in data.items():
                    uid = self.generate_uid_face()
                    new_db[uid] = {
                        "name": name,
                        "embeddings": embeddings,
                        "uid_face": uid
                    }
                self.embeddings_db = new_db
            else:
                self.embeddings_db = data
        except Exception:
            self.embeddings_db = {}

    def detect_faces(self, frame):
        height, width, _ = frame.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        return faces

    def get_face_embedding(self, frame, face):
        aligned_face = self.face_recognizer.alignCrop(frame, face)
        face_feature = self.face_recognizer.feature(aligned_face)
        return face_feature

    def recognize_face(self, frame, face):
        aligned_face = self.face_recognizer.alignCrop(frame, face)
        if self.is_spoof_frame(aligned_face):
            return "Spoof Detected", 0.0, None
        if len(self.embeddings_db) == 0:
            return "Unknown", 0.0, None
        try:
            face_feature = self.get_face_embedding(frame, face)
            best_match = None
            best_score = 0.0
            best_uid = None
            for uid, face_data in self.embeddings_db.items():
                stored_feature = face_data["embeddings"]
                name = face_data["name"]
                norm_face = np.linalg.norm(face_feature)
                norm_stored = np.linalg.norm(stored_feature)
                if norm_face > 0 and norm_stored > 0:
                    score = np.dot(face_feature, stored_feature.T) / (norm_face * norm_stored)
                    score = float(score)
                else:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_match = name
                    best_uid = uid
            if best_score > self.recognition_threshold:
                return best_match, best_score, best_uid
            else:
                return "Unknown", best_score, None
        except Exception:
            return "Error", 0.0, None

    def process_frame(self, frame):
        display_frame = frame.copy()
        faces = self.detect_faces(frame)
        faces_info = []
        if faces is not None:
            for face in faces:
                x, y, w, h, conf = int(face[0]), int(face[1]), int(face[2]), int(face[3]), face[-1]
                name, score, uid_face = self.recognize_face(frame, face)
                faces_info.append({
                    "bbox": (x, y, w, h),
                    "confidence": conf,
                    "name": name,
                    "score": score,
                    "uid_face": uid_face
                })
        return display_frame, faces_info

    def get_best_face_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        best_frame = None
        best_face = None
        best_score = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.detect_faces(frame)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, w, h, conf = int(face[0]), int(face[1]), int(face[2]), int(face[3]), face[-1]
                    face_size = w * h
                    quality_score = conf * face_size
                    if quality_score > best_score:
                        best_score = quality_score
                        best_frame = frame.copy()
                        best_face = face
        cap.release()
        return best_frame, best_face, best_score

    def is_spoof_frame(self, aligned_face, blur_thresh=100.0):
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score < blur_thresh
