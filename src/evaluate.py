"""
evaluate.py  –  CARACTERÍSTICA EXTRA: Evaluación experimental del modelo.

Realiza una evaluación rigurosa con división train/test (80/20) y calcula:
  • Accuracy, Precision, Recall, F1 (macro y por clase)
  • Matriz de confusión (impresa en consola + guardada como PNG)
  • Comparación de umbrales de confianza (CONFIDENCE_THRESHOLD ± variaciones)
  • Curva de precisión vs umbral
  • Tiempo de predicción promedio

Esta información es la base para la sección de Evaluación Experimental del reporte.

Uso:
    python evaluate.py
    python evaluate.py --splits 5      # 5-fold cross-validation
    python evaluate.py --save-plots    # guarda gráficas como PNG
"""

import os
import pickle
import time
import argparse
import random
import csv
import cv2
import numpy as np
from collections import defaultdict

from config import (
    ROSTROS_DIR, CASCADE_PATH, FACE_SIZE,
    LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y,
    CONFIDENCE_THRESHOLD, BASE_DIR
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_all_samples():
    """
    Carga TODAS las muestras disponibles (sin entrenar).
    Retorna list of (face_array, label_id), label_map.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    samples   = []
    label_map = {}
    current_id = 0

    persons = sorted(
        [d for d in os.listdir(ROSTROS_DIR)
         if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
    )

    for person_name in persons:
        person_dir = os.path.join(ROSTROS_DIR, person_name)
        label_map[current_id] = person_name

        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

            # Intentar detectar rostro; si falla, redimensionar directamente
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            else:
                roi = cv2.resize(gray, FACE_SIZE)

            samples.append((roi, current_id))
        current_id += 1

    random.shuffle(samples)
    return samples, label_map


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_f1(y_true, y_pred, n_classes):
    """Calcula métricas macro-promediadas."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        p = tp[c] / (tp[c] + fp[c] + 1e-9)
        r = tp[c] / (tp[c] + fn[c] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return (
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1s),
        precisions, recalls, f1s
    )


def confusion_matrix_manual(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def print_confusion_matrix(cm, label_map):
    names = [label_map[i][:8] for i in range(len(label_map))]
    header = "       " + "  ".join(f"{n:>8}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {names[i]:>6} " + "  ".join(f"{v:>8}" for v in row))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación principal
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(samples, label_map, train_ratio=0.8):
    """Un solo split train/test."""
    n = len(samples)
    split = int(n * train_ratio)
    train_samples = samples[:split]
    test_samples  = samples[split:]

    train_faces  = [s[0] for s in train_samples]
    train_labels = np.array([s[1] for s in train_samples])
    test_faces   = [s[0] for s in test_samples]
    test_labels  = [s[1] for s in test_samples]

    # Entrenar
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,  grid_y=LBPH_GRID_Y
    )
    rec.train(train_faces, train_labels)

    # Predecir
    pred_labels   = []
    confidences   = []
    times         = []

    for face in test_faces:
        t0 = time.perf_counter()
        lbl, conf = rec.predict(face)
        times.append(time.perf_counter() - t0)
        pred_labels.append(lbl if conf < CONFIDENCE_THRESHOLD else -1)
        confidences.append(conf)

    return test_labels, pred_labels, confidences, np.mean(times) * 1000


def run_evaluation(n_splits: int = 1, save_plots: bool = False):
    print("=" * 65)
    print("  SISTEMA UDLAP – Evaluación Experimental del Modelo LBPH")
    print("=" * 65)

    print("\n[1/4] Cargando dataset completo...")
    samples, label_map = load_all_samples()
    n_classes = len(label_map)
    print(f"  Personas   : {n_classes}")
    print(f"  Muestras   : {len(samples)}")

    print(f"\n[2/4] Evaluando con {n_splits} split(s) 80/20...\n")

    all_true, all_pred = [], []
    avg_times = []

    for run in range(n_splits):
        random.shuffle(samples)
        y_true, y_pred, confs, avg_ms = evaluate_split(samples, label_map)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        avg_times.append(avg_ms)

        # Accuracy de este split
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = correct / len(y_true) * 100
        print(f"  Split {run+1}: accuracy={acc:.1f}%  avg_pred={avg_ms:.2f}ms")

    # ── Métricas globales ─────────────────────────────────────────────────────
    print("\n[3/4] Métricas globales:")

    # Filtrar predicciones donde el modelo rechazó (label=-1)
    paired = [(t, p) for t, p in zip(all_true, all_pred) if p != -1]
    if paired:
        y_t, y_p = zip(*paired)
    else:
        y_t, y_p = all_true, all_pred

    correct_total = sum(t == p for t, p in zip(all_true, all_pred))
    accuracy = correct_total / len(all_true) * 100

    # Tasa de rechazo (predicciones con confianza > umbral)
    rejected_pct = sum(1 for p in all_pred if p == -1) / len(all_pred) * 100

    prec, rec, f1, precs_pc, recs_pc, f1s_pc = precision_recall_f1(
        list(y_t), list(y_p), n_classes
    )

    print(f"  Accuracy            : {accuracy:.2f}%")
    print(f"  Precision (macro)   : {prec*100:.2f}%")
    print(f"  Recall    (macro)   : {rec*100:.2f}%")
    print(f"  F1-Score  (macro)   : {f1*100:.2f}%")
    print(f"  Tasa de rechazo     : {rejected_pct:.1f}%")
    print(f"  Tiempo pred. prom.  : {np.mean(avg_times):.2f} ms")

    print("\n  Métricas por persona:")
    print(f"  {'Nombre':<25} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 55)
    for i in range(n_classes):
        print(
            f"  {label_map[i]:<25} "
            f"{precs_pc[i]*100:>9.1f}%"
            f"{recs_pc[i]*100:>7.1f}%"
            f"{f1s_pc[i]*100:>7.1f}%"
        )

    # ── Matriz de confusión ───────────────────────────────────────────────────
    print("\n  Matriz de confusión (filas=real, cols=predicho):")
    cm = confusion_matrix_manual(list(y_t), list(y_p), n_classes)
    print_confusion_matrix(cm, label_map)

    # ── Análisis de umbrales ─────────────────────────────────────────────────
    print("\n[4/4] Análisis de umbrales de confianza:")

    # Re-ejecutar un split con todas las confianzas crudas
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_s = samples[:split];  test_s = samples[split:]
    rec2 = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    rec2.train([s[0] for s in train_s], np.array([s[1] for s in train_s]))

    raw_results = []
    for face, true_lbl in test_s:
        pred_lbl, conf = rec2.predict(face)
        raw_results.append((true_lbl, pred_lbl, conf))

    thresholds = list(range(40, 130, 10))
    print(f"  {'Umbral':>8} {'Accuracy':>10} {'Rechazo':>10}")
    print("  " + "-" * 32)
    threshold_rows = []
    for thr in thresholds:
        preds = [p if c < thr else -1 for (_, p, c) in raw_results]
        trues = [t for (t, _, _) in raw_results]
        acc_t = sum(t == p for t, p in zip(trues, preds)) / len(trues) * 100
        rej_t = sum(1 for p in preds if p == -1) / len(preds) * 100
        mark  = " ◄ actual" if thr == CONFIDENCE_THRESHOLD else ""
        print(f"  {thr:>8} {acc_t:>9.1f}% {rej_t:>9.1f}%{mark}")
        threshold_rows.append({"umbral": thr, "accuracy": acc_t, "rechazo": rej_t})

    # ── Guardar resultados en CSV para el reporte ─────────────────────────────
    results_csv = os.path.join(BASE_DIR, "eval_results.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["umbral", "accuracy", "rechazo"])
        w.writeheader()
        w.writerows(threshold_rows)

    # Guardar métricas generales
    summary_csv = os.path.join(BASE_DIR, "eval_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerows([
            ["accuracy",          f"{accuracy:.4f}"],
            ["precision_macro",   f"{prec:.4f}"],
            ["recall_macro",      f"{rec:.4f}"],
            ["f1_macro",          f"{f1:.4f}"],
            ["tasa_rechazo",      f"{rejected_pct:.4f}"],
            ["avg_pred_ms",       f"{np.mean(avg_times):.4f}"],
        ])

    print(f"\n  Resultados guardados en:")
    print(f"    {results_csv}")
    print(f"    {summary_csv}")

    # ── Gráficas (requiere matplotlib) ───────────────────────────────────────
    if save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            thr_vals = [r["umbral"]   for r in threshold_rows]
            acc_vals = [r["accuracy"] for r in threshold_rows]
            rej_vals = [r["rechazo"]  for r in threshold_rows]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle("Evaluación del Modelo LBPH – UDLAP", fontsize=13)

            # Curva accuracy vs umbral
            axes[0].plot(thr_vals, acc_vals, "b-o", label="Accuracy")
            axes[0].axvline(CONFIDENCE_THRESHOLD, color="r", linestyle="--",
                            label=f"Umbral={CONFIDENCE_THRESHOLD}")
            axes[0].set_title("Accuracy vs Umbral")
            axes[0].set_xlabel("Umbral de confianza LBPH")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].legend()
            axes[0].grid(True)

            # Matriz de confusión
            im = axes[1].imshow(cm, cmap="Blues")
            axes[1].set_title("Matriz de Confusión")
            axes[1].set_xlabel("Predicho")
            axes[1].set_ylabel("Real")
            short_names = [label_map[i][:10] for i in range(n_classes)]
            axes[1].set_xticks(range(n_classes))
            axes[1].set_yticks(range(n_classes))
            axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
            axes[1].set_yticklabels(short_names, fontsize=8)
            for i in range(n_classes):
                for j in range(n_classes):
                    axes[1].text(j, i, str(cm[i, j]),
                                 ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=axes[1])

            plt.tight_layout()
            plot_path = os.path.join(BASE_DIR, "eval_plots.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"    {plot_path}")
            plt.close()
        except ImportError:
            print("  [WARN] matplotlib no instalado. Omitiendo gráficas.")

    print("\n✓ Evaluación completada.\n")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación del modelo – UDLAP")
    parser.add_argument("--splits",     type=int, default=1,
                        help="Número de splits aleatorios (default: 1)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Guarda gráficas PNG (requiere matplotlib)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(n_splits=args.splits, save_plots=args.save_plots)
