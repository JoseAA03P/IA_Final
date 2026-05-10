"""
train.py  –  Entrena el modelo LBPH con los rostros en ROSTROS_DIR.

Estructura esperada de carpetas:
    Rostros/
        AlumnoA/
            img1.jpg
            img2.jpg
            ...
        AlumnoB/
            img1.jpg
            ...

Uso:
    python train.py
"""

import os
import pickle
import cv2
import numpy as np
from config import (
    ROSTROS_DIR, MODEL_DIR, MODEL_PATH, LABELS_PATH,
    CASCADE_PATH, FACE_SIZE,
    LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(image_gray, detector):
    """
    Detecta el rostro más grande en la imagen y lo recorta.
    Retorna (roi_gray, bbox) o (None, None) si no se encontró.
    """
    faces = detector.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None

    # Tomar el rostro de mayor área (más cercano a la cámara)
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    roi = cv2.resize(image_gray[y:y+h, x:x+w], FACE_SIZE)
    return roi, (x, y, w, h)


def load_dataset(detector):
    """
    Recorre ROSTROS_DIR y carga imágenes con sus etiquetas numéricas.
    Retorna (faces_list, labels_list, label_map: {int -> nombre}).
    """
    faces, labels = [], []
    label_map   = {}       # {id_numerico: nombre_persona}
    current_id  = 0

    if not os.path.isdir(ROSTROS_DIR):
        raise FileNotFoundError(
            f"No se encontró la carpeta de rostros: {ROSTROS_DIR}\n"
            "Crea la carpeta y agrega subcarpetas con el nombre de cada persona."
        )

    persons = sorted(
        [d for d in os.listdir(ROSTROS_DIR)
         if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
    )

    if not persons:
        raise ValueError(
            f"La carpeta {ROSTROS_DIR} no contiene subcarpetas de personas."
        )

    for person_name in persons:
        person_dir = os.path.join(ROSTROS_DIR, person_name)
        label_map[current_id] = person_name
        count = 0

        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        for fname in image_files:
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] No se pudo leer: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Ecualización de histograma para mejorar iluminación variable
            gray = cv2.equalizeHist(gray)

            face_roi, _ = detect_and_crop_face(gray, detector)
            if face_roi is not None:
                faces.append(face_roi)
                labels.append(current_id)
                count += 1
            else:
                # Si la imagen ya es solo el rostro recortado (sin contexto)
                resized = cv2.resize(gray, FACE_SIZE)
                faces.append(resized)
                labels.append(current_id)
                count += 1

        print(f"  [{current_id:02d}] {person_name:20s}  →  {count} imágenes cargadas")
        current_id += 1

    return faces, labels, label_map


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL – UDLAP")
    print("  Módulo: Entrenamiento del modelo LBPH")
    print("=" * 60)

    # Crear carpeta de modelos si no existe
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cargar detector de rostros Haar Cascade
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el cascade: {CASCADE_PATH}")

    # ── Cargar dataset ────────────────────────────────────────────────────────
    print(f"\n[1/3] Cargando imágenes desde: {ROSTROS_DIR}\n")
    faces, labels, label_map = load_dataset(detector)

    total_images = len(faces)
    total_persons = len(label_map)
    print(f"\n  Total personas : {total_persons}")
    print(f"  Total imágenes : {total_images}")

    if total_images < 2:
        raise ValueError("Se necesitan al menos 2 imágenes para entrenar.")

    # ── Crear y entrenar reconocedor LBPH ─────────────────────────────────────
    print("\n[2/3] Entrenando modelo LBPH...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )
    recognizer.train(faces, np.array(labels))

    # ── Guardar modelo y etiquetas ────────────────────────────────────────────
    print("[3/3] Guardando modelo...")
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"\n  Modelo guardado  → {MODEL_PATH}")
    print(f"  Etiquetas        → {LABELS_PATH}")

    # ── Evaluación rápida (leave-one-out simplificado) ────────────────────────
    print("\n[INFO] Evaluación rápida sobre el conjunto de entrenamiento:")
    correct = 0
    for face, true_label in zip(faces, labels):
        pred_label, confidence = recognizer.predict(face)
        if pred_label == true_label:
            correct += 1

    accuracy = correct / total_images * 100
    print(f"  Precisión (train-set): {accuracy:.1f}%  ({correct}/{total_images})")
    print(
        "\n  NOTA: Para una evaluación más rigurosa, divide el dataset en\n"
        "  train/test antes de entrenar (ver demo.py para umbral de confianza)."
    )

    print("\n✓ Entrenamiento completado exitosamente.\n")
    return recognizer, label_map


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
