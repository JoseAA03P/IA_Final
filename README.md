# Sistema de Reconocimiento Facial – Control de Acceso UDLAP
**Inteligencia Artificial · Primavera 2026**  
Integrantes: Dana · Gil · José Ángel Palomares  
Profesor: Dr. Edwin Montes Orozco

---

## Descripción

Agente inteligente que controla el acceso a instalaciones universitarias mediante reconocimiento facial en tiempo real. El agente percibe un flujo de video, detecta rostros con **Haar Cascade**, los reconoce con **LBPH** y toma una decisión binaria: **Aprobado** o **No aprobado**, registrando cada evento automáticamente.

---

## Estructura del repositorio

```
proyecto/
├── src/
│   ├── config.py        # Parámetros globales y rutas
│   ├── train.py         # Entrenamiento del modelo LBPH
│   ├── register.py      # Captura de muestras con filtros de calidad
│   ├── demo.py          # Demo en tiempo real (pantalla principal)
│   ├── evaluate.py      # Métricas: accuracy, F1, matriz de confusión
│   ├── access_log.py    # Registro CSV de eventos de acceso
│   └── main.py          # Menú principal
├── data/
│   └── Rostros/         # Subcarpeta por persona con sus imágenes
├── experiments/
│   └── eval_results.csv # Análisis de umbrales (generado por evaluate.py)
├── results/
│   ├── eval_summary.csv # Métricas finales (generado por evaluate.py)
│   └── eval_plots.png   # Gráficas (generado por evaluate.py --save-plots)
├── requirements.txt
└── README.md
```

---

## Dependencias

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

> **Importante:** `opencv-contrib-python` es obligatorio (incluye el módulo LBPH).

Python 3.9+. No requiere GPU.

---

## Cómo ejecutar el sistema

### 1. Preparar el dataset
Crear subcarpetas dentro de `data/Rostros/` con el nombre de cada persona y colocar sus imágenes adentro:
```
data/Rostros/
├── Jose_Angel/
│   ├── 0001.jpg
│   └── ...
└── Otra_Persona/
    └── ...
```

### 2. Entrenar el modelo
```bash
cd src
python train.py
```

### 3. Registrar nuevos usuarios (webcam)
```bash
python register.py
```

### 4. Ejecutar la demo en tiempo real
```bash
python demo.py
# Opciones:
python demo.py --camera 1 --threshold 55
```

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `s` | Mostrar estadísticas |
| `r` | Guardar screenshot |

### 5. Evaluar el modelo
```bash
python evaluate.py --splits 5 --save-plots
```
Genera `results/eval_summary.csv` y `results/eval_plots.png`.

### 6. Menú interactivo
```bash
python main.py
```

---

## Ejemplo de ejecución

```
$ python train.py
======================================================
  SISTEMA DE RECONOCIMIENTO FACIAL – UDLAP
  Módulo: Entrenamiento del modelo LBPH
======================================================

[1/3] Cargando imágenes desde: data/Rostros

  [00] Jose_Angel_Palomares    →  50 imágenes cargadas

  Total personas : 1
  Total imágenes : 50

[2/3] Entrenando modelo LBPH...
[3/3] Guardando modelo...

  Precisión (train-set): 100.0%  (50/50)
✓ Entrenamiento completado exitosamente.
```

```
$ python demo.py
  Modelo cargado. Personas registradas: 1
    [0] Jose_Angel_Palomares
  [CAM] Backend: DirectShow (Windows) OK ✓
  Resolución: 1280x720   Umbral: 55
```

---

## Parámetros clave (`config.py`)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `CONFIDENCE_THRESHOLD` | `55` | Distancia LBPH máxima para aprobar |
| `FACE_SIZE` | `(200, 200)` | Normalización de imagen |
| `BLUR_THRESHOLD` | `120` | Nitidez mínima al registrar |
| `MIN_FACE_RATIO` | `0.18` | Tamaño mínimo del rostro en el frame |
