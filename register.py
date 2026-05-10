"""
register.py  –  Registra nuevos rostros capturando imágenes con la webcam.

MEJORAS v2:
  1. Filtro de nitidez (Laplacian variance): descarta frames borrosos.
  2. Filtro de tamaño mínimo: el rostro debe ser suficientemente grande.
  3. Filtro de centrado: el rostro debe estar cerca del centro del frame.
  4. Indicador de calidad en pantalla (barra + texto).
  5. Guía de poses: pide al usuario girar levemente la cabeza
     para capturar variedad y mejorar la robustez del modelo.
  6. Intervalo mínimo entre capturas para evitar frames casi idénticos.

Uso:
    python register.py
    python register.py --name "Juan Perez" --samples 50 --retrain
"""

import os
import sys
import cv2
import argparse
import time
import numpy as np

from config import (
    ROSTROS_DIR, CASCADE_PATH, FACE_SIZE, MIN_SAMPLES,
    BLUR_THRESHOLD, MIN_FACE_RATIO, FACE_CENTER_MARGIN, CAPTURE_INTERVAL,
    COLOR_APPROVED, COLOR_REJECTED, COLOR_INFO, COLOR_WARNING
)


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de calidad de imagen
# ─────────────────────────────────────────────────────────────────────────────

def sharpness_score(gray_roi) -> float:
    """
    Calcula la nitidez usando la varianza del Laplaciano.
    Valores altos = imagen nítida. Valores bajos = borrosa o en movimiento.
    """
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


def check_face_quality(face_x, face_y, face_w, face_h,
                        frame_w, frame_h, gray_roi) -> tuple[bool, list[str]]:
    """
    Evalúa la calidad del rostro detectado.

    Retorna:
        (es_valido: bool, razones_de_rechazo: list[str])
    """
    issues = []

    # ── 1. Nitidez ────────────────────────────────────────────────────────────
    blur = sharpness_score(gray_roi)
    if blur < BLUR_THRESHOLD:
        issues.append(f"Imagen borrosa ({blur:.0f} < {BLUR_THRESHOLD:.0f})")

    # ── 2. Tamaño mínimo del rostro ───────────────────────────────────────────
    face_ratio_w = face_w / frame_w
    face_ratio_h = face_h / frame_h
    if face_ratio_w < MIN_FACE_RATIO or face_ratio_h < MIN_FACE_RATIO:
        issues.append(f"Rostro muy pequeno ({face_ratio_w*100:.0f}% ancho) — acercate")

    # ── 3. Centrado del rostro ────────────────────────────────────────────────
    center_x = (face_x + face_w / 2) / frame_w   # 0-1
    center_y = (face_y + face_h / 2) / frame_h   # 0-1
    off_x = abs(center_x - 0.5)
    off_y = abs(center_y - 0.5)
    if off_x > FACE_CENTER_MARGIN or off_y > FACE_CENTER_MARGIN:
        issues.append("Centra tu rostro en la camara")

    return (len(issues) == 0), issues


# ─────────────────────────────────────────────────────────────────────────────
# Apertura de cámara (mismo sistema que demo.py)
# ─────────────────────────────────────────────────────────────────────────────

def open_camera(camera_id: int = 0):
    backends = []
    if sys.platform == "win32":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF,  "Media Foundation"),
            (cv2.CAP_ANY,   "Auto"),
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY,  "Auto"),
        ]

    for backend, name in backends:
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                print(f"  [CAM] Backend: {name} OK ✓")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
                for _ in range(5):
                    cap.read()
                return cap
            cap.release()

    raise RuntimeError("No se pudo acceder a la cámara.")


# ─────────────────────────────────────────────────────────────────────────────
# Guía de poses
# ─────────────────────────────────────────────────────────────────────────────

# Secuencia de instrucciones que se muestran a medida que avanza la captura.
# Cada entrada es (porcentaje_inicio, texto_instruccion).
POSE_GUIDE = [
    (0.00, "Mira directamente a la camara"),
    (0.20, "Gira levemente a la DERECHA"),
    (0.40, "Gira levemente a la IZQUIERDA"),
    (0.60, "Inclina la cabeza hacia ARRIBA"),
    (0.80, "Mira directamente (expresion neutral)"),
]

def get_pose_instruction(progress: float) -> str:
    """Devuelve la instrucción de pose según el progreso (0.0 – 1.0)."""
    instruction = POSE_GUIDE[0][1]
    for threshold, text in POSE_GUIDE:
        if progress >= threshold:
            instruction = text
    return instruction


# ─────────────────────────────────────────────────────────────────────────────
# Captura principal
# ─────────────────────────────────────────────────────────────────────────────

def capture_faces(name: str, n_samples: int = MIN_SAMPLES, camera_id: int = 0):
    """
    Abre la webcam y guarda n_samples imágenes de calidad en ROSTROS_DIR/<name>/.

    Solo guarda un frame cuando pasa TODOS los filtros de calidad:
      ✓ nitidez (no borroso)
      ✓ tamaño suficiente (cerca de la cámara)
      ✓ rostro centrado
    """
    save_dir = os.path.join(ROSTROS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    existing = [f for f in os.listdir(save_dir)
                if f.lower().endswith((".jpg", ".png"))]
    offset = len(existing)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el cascade: {CASCADE_PATH}")

    cap = open_camera(camera_id)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Registrando : {name}")
    print(f"  Destino     : {save_dir}")
    print(f"  Muestras    : {n_samples}  (solo se guardan frames de calidad)")
    print(f"  Filtros     : blur>{BLUR_THRESHOLD}  |  "
          f"face>{MIN_FACE_RATIO*100:.0f}% frame  |  centrado")
    print("  Presiona 'q' para cancelar.\n")

    count      = 0
    rejected   = 0
    last_saved = time.time()

    while count < n_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        display  = frame.copy()
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq  = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        # ── Sin rostro detectado ──────────────────────────────────────────────
        if len(faces) == 0:
            _draw_no_face(display, frame_w, frame_h)

        else:
            # Tomar el rostro más grande
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            roi_gray = gray_eq[y:y+h, x:x+w]

            valid, issues = check_face_quality(x, y, w, h,
                                               frame_w, frame_h, roi_gray)

            progress    = count / n_samples
            pose_text   = get_pose_instruction(progress)
            blur_val    = sharpness_score(roi_gray)

            # ── Guardar si pasa todos los filtros ─────────────────────────────
            if valid and (time.time() - last_saved) >= CAPTURE_INTERVAL:
                roi_resized = cv2.resize(roi_gray, FACE_SIZE)
                filename = os.path.join(save_dir, f"{offset + count:04d}.jpg")
                cv2.imwrite(filename, roi_resized)
                count      += 1
                last_saved  = time.time()

            elif not valid:
                rejected += 1

            # ── Dibujar overlay del rostro ────────────────────────────────────
            color = COLOR_APPROVED if valid else COLOR_WARNING
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            _draw_quality_card(display, x, y, w, h,
                               valid, issues, blur_val)

        # ── HUD de progreso ───────────────────────────────────────────────────
        _draw_progress_hud(display, name, count, n_samples,
                           rejected, pose_text, frame_w)

        cv2.imshow("Registro de Rostros – UDLAP  |  'q' cancela", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  [INFO] Registro cancelado.")
            break

    cap.release()
    cv2.destroyAllWindows()

    total = offset + count
    print(f"\n  ✓ {count} imágenes nuevas guardadas  (total en carpeta: {total})")
    print(f"    Frames rechazados por calidad: {rejected}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de dibujado
# ─────────────────────────────────────────────────────────────────────────────

def _draw_no_face(display, fw, fh):
    """Aviso cuando no se detecta ningún rostro."""
    cv2.putText(display, "Rostro no detectado — acercate y centra tu cara",
                (20, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_REJECTED, 2, cv2.LINE_AA)


def _draw_quality_card(display, x, y, w, h, valid, issues, blur_val):
    """Tarjeta de calidad encima del rectángulo del rostro."""
    lines = []
    color = COLOR_APPROVED if valid else COLOR_WARNING

    if valid:
        lines.append(("OK – calidad suficiente", COLOR_APPROVED))
    else:
        for iss in issues:
            lines.append((iss, COLOR_WARNING))

    lines.append((f"Nitidez: {blur_val:.0f}  (min {BLUR_THRESHOLD:.0f})",
                  (200, 200, 200)))

    card_h = len(lines) * 20 + 10
    tag_y  = max(y - card_h - 4, 0)
    ov     = display.copy()
    cv2.rectangle(ov, (x, tag_y), (x + max(w, 260), tag_y + card_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, display, 0.35, 0, display)

    for i, (text, col) in enumerate(lines):
        cv2.putText(display, text,
                    (x + 5, tag_y + 18 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def _draw_progress_hud(display, name, count, n_samples,
                       rejected, pose_text, frame_w):
    """Barra de progreso y guía de pose en la parte inferior."""
    fh = display.shape[0]
    bar_y   = fh - 80
    bar_x   = 10
    bar_w   = frame_w - 20
    bar_h   = 18
    pct     = count / n_samples
    filled  = int(bar_w * pct)

    # Fondo semitransparente
    ov = display.copy()
    cv2.rectangle(ov, (0, bar_y - 30), (frame_w, fh), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, display, 0.45, 0, display)

    # Instrucción de pose
    cv2.putText(display, f"Pose: {pose_text}",
                (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_INFO, 2, cv2.LINE_AA)

    # Barra de progreso
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  COLOR_APPROVED, -1)

    # Texto sobre la barra
    pct_text = f"{name}  –  {count}/{n_samples}  ({pct*100:.0f}%)   rechazados: {rejected}"
    cv2.putText(display, pct_text,
                (bar_x, bar_y + bar_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Menú interactivo
# ─────────────────────────────────────────────────────────────────────────────

def interactive_menu():
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL – UDLAP")
    print("  Módulo: Registro de Nuevos Usuarios  (v2 con filtros de calidad)")
    print("=" * 60)

    name = input("\nNombre completo del usuario a registrar: ").strip()
    if not name:
        print("[ERROR] El nombre no puede estar vacío.")
        return

    samples_str = input(f"Número de muestras [{MIN_SAMPLES}]: ").strip()
    try:
        n_samples = int(samples_str) if samples_str else MIN_SAMPLES
    except ValueError:
        n_samples = MIN_SAMPLES

    cam_str = input("Índice de cámara [0]: ").strip()
    try:
        cam = int(cam_str) if cam_str else 0
    except ValueError:
        cam = 0

    count = capture_faces(name, n_samples, cam)

    if count > 0:
        retrain = input("\n¿Re-entrenar el modelo ahora? (s/n): ").strip().lower()
        if retrain == "s":
            from train import train
            train()
        else:
            print("[INFO] Re-entrenamiento omitido. Ejecuta train.py cuando quieras.")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Registro de rostros – UDLAP v2")
    parser.add_argument("--name",    type=str)
    parser.add_argument("--samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.name:
        count = capture_faces(args.name, args.samples, args.camera)
        if count > 0 and args.retrain:
            from train import train
            train()
    else:
        interactive_menu()