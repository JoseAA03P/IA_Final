"""
Configuración global del sistema de reconocimiento facial – UDLAP
Modifica estas rutas si cambias la ubicación del proyecto.
"""

import os

# ── Rutas base ────────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\japal\Downloads\ia\Final"
ROSTROS_DIR     = os.path.join(BASE_DIR, "Rostros")
MODEL_DIR       = os.path.join(BASE_DIR, "Models")
LOG_FILE        = os.path.join(BASE_DIR, "access_log.csv")

# ── Archivos del modelo ───────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(MODEL_DIR, "lbph_model.xml")
LABELS_PATH     = os.path.join(MODEL_DIR, "labels.pkl")

# ── Haar Cascade (viene con OpenCV) ──────────────────────────────────────────
import cv2
CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Parámetros de reconocimiento ─────────────────────────────────────────────
# LBPH: distancia MENOR = mejor coincidencia.
# CONFIDENCE_THRESHOLD bajo → más estricto → menos falsos positivos.
#
#   < 50  →  prácticamente idéntico
#   50-60 →  muy seguro (recomendado producción)   ← NUEVO VALOR
#   60-80 →  moderado
#   > 80  →  permisivo (muchos falsos positivos)
#
CONFIDENCE_THRESHOLD = 55   # era 80; ahora solo aprueba coincidencias sólidas

# Tamaño al que se normalizan los rostros antes de entrenar / predecir
FACE_SIZE       = (200, 200)

# Mínimo de imágenes requeridas por persona al registrar (via webcam)
MIN_SAMPLES     = 50   # era 30; más muestras → modelo más robusto

# ── Filtros de calidad para el registro ──────────────────────────────────────
# Varianza del Laplaciano: mide nitidez. Frames por debajo de este valor
# se descartan automáticamente (imagen borrosa / movimiento).
BLUR_THRESHOLD          = 120.0   # cuanto mayor, más exigente

# El rostro detectado debe ocupar al menos este % del ancho y alto del frame.
# Evita registrar rostros muy pequeños / lejanos.
MIN_FACE_RATIO          = 0.18    # 18 % del ancho o alto del frame

# El centro del rostro debe estar dentro de este margen central del frame
# (0.5 ± FACE_CENTER_MARGIN). Evita rostros en los bordes.
FACE_CENTER_MARGIN      = 0.30    # ± 30 % alrededor del centro

# Pausa mínima entre capturas (segundos). Da tiempo a que el modelo reciba
# poses ligeramente distintas en lugar de frames casi idénticos.
CAPTURE_INTERVAL        = 0.25    # ≈ 4 imágenes por segundo como máximo

# ── Parámetros LBPH ───────────────────────────────────────────────────────────
LBPH_RADIUS     = 1
LBPH_NEIGHBORS  = 8
LBPH_GRID_X     = 8
LBPH_GRID_Y     = 8

# ── Colores BGR ───────────────────────────────────────────────────────────────
COLOR_APPROVED  = (0,   220,   0)    # verde
COLOR_REJECTED  = (0,   0,   220)    # rojo
COLOR_INFO      = (220, 220,   0)    # amarillo
COLOR_WARNING   = (0,   165, 255)    # naranja