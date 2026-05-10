"""
main.py  –  Menú principal del Sistema de Reconocimiento Facial UDLAP.

Uso:
    python main.py
"""

import os
import sys


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      SISTEMA DE RECONOCIMIENTO FACIAL – UDLAP                ║
║      Control de Acceso Inteligente                           ║
║      Inteligencia Artificial  •  Primavera 2026              ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
  [1]  Entrenar modelo         (train.py)
  [2]  Registrar nuevo usuario (register.py)
  [3]  Demo en tiempo real     (demo.py)
  [4]  Evaluar modelo          (evaluate.py)
  [5]  Ver log de accesos      (access_log.py)
  [6]  Configuración actual    (config.py)
  [0]  Salir
"""


def show_config():
    from config import (
        ROSTROS_DIR, MODEL_PATH, LOG_FILE,
        CONFIDENCE_THRESHOLD, FACE_SIZE, MIN_SAMPLES
    )
    print("\n  ── Configuración actual ──────────────────────────────────")
    print(f"  Carpeta de rostros : {ROSTROS_DIR}")
    print(f"  Modelo             : {MODEL_PATH}")
    print(f"  Log de accesos     : {LOG_FILE}")
    print(f"  Umbral confianza   : {CONFIDENCE_THRESHOLD}")
    print(f"  Tamaño de imagen   : {FACE_SIZE}")
    print(f"  Muestras mínimas   : {MIN_SAMPLES}")

    # Contar personas y muestras
    try:
        persons = [
            d for d in os.listdir(ROSTROS_DIR)
            if os.path.isdir(os.path.join(ROSTROS_DIR, d))
        ]
        total_imgs = sum(
            len([
                f for f in os.listdir(os.path.join(ROSTROS_DIR, p))
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            for p in persons
        )
        print(f"\n  Personas registradas  : {len(persons)}")
        print(f"  Total de imágenes     : {total_imgs}")
        for p in persons:
            n = len([
                f for f in os.listdir(os.path.join(ROSTROS_DIR, p))
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            print(f"    • {p:<25}  {n} imágenes")
    except FileNotFoundError:
        print("  [WARN] Carpeta de rostros no encontrada aún.")
    print()


def main():
    print(BANNER)

    while True:
        print(MENU)
        choice = input("  Selecciona una opción: ").strip()

        if choice == "1":
            from train import train
            train()

        elif choice == "2":
            from register import interactive_menu
            interactive_menu()

        elif choice == "3":
            cam = input("  Índice de cámara [0]: ").strip()
            cam = int(cam) if cam.isdigit() else 0
            thr = input(f"  Umbral de confianza [80]: ").strip()
            thr = int(thr) if thr.isdigit() else 80
            from demo import run_demo
            run_demo(camera_id=cam, threshold=thr)

        elif choice == "4":
            splits = input("  Número de splits [1]: ").strip()
            splits = int(splits) if splits.isdigit() else 1
            plots  = input("  Guardar gráficas PNG? (s/n) [n]: ").strip().lower()
            from evaluate import run_evaluation
            run_evaluation(n_splits=splits, save_plots=(plots == "s"))

        elif choice == "5":
            from access_log import print_summary
            print_summary()

        elif choice == "6":
            show_config()

        elif choice == "0":
            print("\n  ¡Hasta luego!\n")
            sys.exit(0)

        else:
            print("  [ERROR] Opción no válida.\n")


if __name__ == "__main__":
    main()
