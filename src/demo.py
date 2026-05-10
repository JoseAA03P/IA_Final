"""
demo.py  –  Pantalla de reconocimiento facial en tiempo real.
            FIX: backend DirectShow para Windows (pantalla negra resuelta).

Sobre cada rostro detectado muestra:
  • Nombre reconocido  (o "Desconocido")
  • Nivel de confianza (barra visual)
  • "Aprobado" (verde) / "No aprobado" (rojo)

Controles durante la demo:
    q  –  Salir
    s  –  Mostrar estadísticas en consola
    r  –  Capturar screenshot

Uso:
    python demo.py
    python demo.py --threshold 75 --camera 0
"""

import os
import pickle
import time
import argparse
import sys
import cv2
import numpy as np

from config import (
    MODEL_PATH, LABELS_PATH, CASCADE_PATH, FACE_SIZE,
    CONFIDENCE_THRESHOLD,
    COLOR_APPROVED, COLOR_REJECTED, COLOR_INFO
)
from access_log import log_event, print_summary


# ─────────────────────────────────────────────────────────────────────────────
# Abrir cámara (con diagnóstico y fallback de backend)
# ─────────────────────────────────────────────────────────────────────────────

def open_camera(camera_id: int = 0):
    """
    Intenta abrir la cámara con distintos backends en orden de preferencia.
    En Windows el backend DirectShow (CAP_DSHOW) suele ser necesario para
    evitar la pantalla negra.
    """
    backends = []

    if sys.platform == "win32":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_MSMF,  "Media Foundation (Windows)"),
            (cv2.CAP_ANY,   "Auto"),
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2 (Linux)"),
            (cv2.CAP_ANY,  "Auto"),
        ]

    for backend, name in backends:
        print(f"  [CAM] Probando backend: {name} ...", end=" ", flush=True)
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            # Verificar que realmente entrega frames (no solo que "abre")
            ok, test_frame = cap.read()
            if ok and test_frame is not None and test_frame.size > 0:
                print("OK ✓")
                # Configurar resolución DESPUES de confirmar que funciona
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Calentar la cámara con unos frames adicionales
                for _ in range(5):
                    cap.read()
                return cap
            else:
                print("abre pero sin frames, descartando.")
                cap.release()
        else:
            print("no disponible.")

    raise RuntimeError(
        f"\n[ERROR] No se pudo abrir la cámara {camera_id} con ningún backend.\n"
        "Verifica que:\n"
        "  1. La cámara no esté usada por otra app (Teams, Zoom, OBS, etc.)\n"
        "  2. Tienes permisos: Configuración → Privacidad → Cámara\n"
        "  3. El índice sea correcto (prueba --camera 1 o --camera 2)\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            "Modelo no encontrado. Ejecuta primero:  python train.py"
        )
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)
    return recognizer, label_map


# ─────────────────────────────────────────────────────────────────────────────
# Dibujado de overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_face_overlay(frame, x, y, w, h, name, confidence, approved, threshold):
    color = COLOR_APPROVED if approved else COLOR_REJECTED
    label = "Aprobado" if approved else "No aprobado"

    # Recuadro del rostro
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Tarjeta encima del rostro (fondo semitransparente)
    tag_h = 60
    tag_y = max(y - tag_h - 4, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, tag_y), (x + w, tag_y + tag_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, name,
                (x + 6, tag_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label,
                (x + 6, tag_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)

    # Barra de confianza debajo del rectángulo
    bar_y    = y + h + 8
    bar_h    = 10
    conf_pct = max(0.0, 1.0 - confidence / max(threshold * 2, 1))
    filled_w = int(w * conf_pct)
    cv2.rectangle(frame, (x, bar_y), (x + w,        bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, bar_y), (x + filled_w, bar_y + bar_h), color,        -1)
    cv2.putText(frame,
                f"Conf: {conf_pct*100:.0f}%  dist={confidence:.1f}",
                (x, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)


def draw_hud(frame, fps, total_events, threshold):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    lines = [
        ("UDLAP - Control de Acceso", COLOR_INFO),
        (f"FPS: {fps:5.1f}   Umbral: {threshold}", (200, 200, 200)),
        (f"Eventos registrados: {total_events}",    (200, 200, 200)),
        ("q=salir  s=stats  r=screenshot",           (160, 160, 160)),
    ]
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown de log (evitar registrar el mismo evento cada frame)
# ─────────────────────────────────────────────────────────────────────────────

_last_logged = {}
COOLDOWN_SEC = 3.0


def should_log(key: str) -> bool:
    now = time.time()
    if key not in _last_logged or now - _last_logged[key] > COOLDOWN_SEC:
        _last_logged[key] = now
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(camera_id: int = 0, threshold: int = CONFIDENCE_THRESHOLD):
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL - UDLAP")
    print("  Modulo: Demo en tiempo real")
    print("=" * 60)

    # Verificar modelo
    recognizer, label_map = load_model()
    print(f"  Modelo cargado. Personas registradas: {len(label_map)}")
    for lid, name in label_map.items():
        print(f"    [{lid}] {name}")

    # Cargar detector Haar Cascade
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el cascade: {CASCADE_PATH}")

    # Abrir cámara con diagnóstico automático
    print(f"\n  Abriendo camara {camera_id}...")
    cap = open_camera(camera_id)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolucion: {real_w}x{real_h}")
    print(f"  Umbral de confianza: {threshold}")
    print("\n  [INFO] Controles: q=salir  s=estadisticas  r=screenshot\n")

    total_events = 0
    fps_counter  = 0
    fps_start    = time.time()
    fps_display  = 0.0
    screenshot_n = 0
    empty_frames = 0

    while True:
        ret, frame = cap.read()

        # Manejar frames vacios en lugar de colgar
        if not ret or frame is None or frame.size == 0:
            empty_frames += 1
            if empty_frames % 30 == 0:
                print(f"[WARN] {empty_frames} frames vacios consecutivos.")
            wait_screen = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(wait_screen, "Esperando camara...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_INFO, 2)
            cv2.imshow("Control de Acceso UDLAP  -  'q' para salir", wait_screen)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        empty_frames = 0

        # Preprocesamiento
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Deteccion de rostros
        faces = detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray_eq[y:y+h, x:x+w], FACE_SIZE)
            label_id, confidence = recognizer.predict(roi)

            approved = confidence < threshold
            name     = label_map.get(label_id, "Desconocido") if approved else "Desconocido"

            log_key = name if approved else f"__unk_{x}"
            if should_log(log_key):
                log_event(name, confidence, approved)
                total_events += 1

            draw_face_overlay(frame, x, y, w, h, name, confidence, approved, threshold)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        # HUD global
        draw_hud(frame, fps_display, total_events, threshold)

        cv2.imshow("Control de Acceso UDLAP  -  'q' para salir", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print_summary()
        elif key == ord("r"):
            fname = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            screenshot_n += 1
            print(f"[INFO] Screenshot guardado: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Demo finalizada.")
    print_summary()


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Demo reconocimiento facial UDLAP")
    parser.add_argument("--camera",    type=int, default=0,
                        help="Indice de camara (default: 0)")
    parser.add_argument("--threshold", type=int, default=CONFIDENCE_THRESHOLD,
                        help=f"Umbral LBPH (default: {CONFIDENCE_THRESHOLD})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(camera_id=args.camera, threshold=args.threshold)