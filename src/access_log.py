"""
access_log.py  –  Registro persistente de intentos de acceso.

Cada fila del CSV contiene:
    timestamp, nombre, confianza, estado (Aprobado / No aprobado)

Funciones exportadas:
    log_event(name, confidence, approved)
    print_summary()
    get_stats() -> dict
"""

import csv
import os
from datetime import datetime
from config import LOG_FILE


# ─────────────────────────────────────────────────────────────────────────────
# Escritura
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ["timestamp", "nombre", "confianza", "estado"]

def _ensure_header():
    """Crea el archivo con encabezado si no existe."""
    if not os.path.isfile(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_event(name: str, confidence: float, approved: bool):
    """
    Registra un evento de acceso en el CSV.

    Args:
        name       : nombre reconocido (o "Desconocido")
        confidence : valor de confianza LBPH (distancia; menor = mejor)
        approved   : True si el acceso fue concedido
    """
    _ensure_header()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nombre":    name,
        "confianza": f"{confidence:.2f}",
        "estado":    "Aprobado" if approved else "No aprobado",
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura y estadísticas
# ─────────────────────────────────────────────────────────────────────────────

def _read_log():
    if not os.path.isfile(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_stats() -> dict:
    """
    Calcula estadísticas del log.

    Retorna dict con:
        total, approved, rejected, approval_rate,
        by_person: {nombre: {total, approved, avg_confidence}}
    """
    rows = _read_log()
    if not rows:
        return {}

    total    = len(rows)
    approved = sum(1 for r in rows if r["estado"] == "Aprobado")
    rejected = total - approved

    by_person = {}
    for r in rows:
        n = r["nombre"]
        if n not in by_person:
            by_person[n] = {"total": 0, "approved": 0, "conf_sum": 0.0}
        by_person[n]["total"]    += 1
        by_person[n]["approved"] += 1 if r["estado"] == "Aprobado" else 0
        by_person[n]["conf_sum"] += float(r["confianza"])

    # Calcular promedio de confianza por persona
    for n, d in by_person.items():
        d["avg_confidence"] = d["conf_sum"] / d["total"] if d["total"] else 0

    return {
        "total":         total,
        "approved":      approved,
        "rejected":      rejected,
        "approval_rate": approved / total * 100 if total else 0,
        "by_person":     by_person,
    }


def print_summary():
    """Imprime un resumen formateado del registro de accesos."""
    stats = get_stats()
    if not stats:
        print("[INFO] El log de accesos está vacío.")
        return

    print("\n" + "=" * 60)
    print("  RESUMEN DE ACCESOS – SISTEMA UDLAP")
    print("=" * 60)
    print(f"  Total de intentos : {stats['total']}")
    print(f"  Aprobados         : {stats['approved']}")
    print(f"  Rechazados        : {stats['rejected']}")
    print(f"  Tasa de aprobación: {stats['approval_rate']:.1f}%")
    print("\n  Por persona:")
    print(f"  {'Nombre':<25} {'Intentos':>9} {'Aprobados':>10} {'Conf. Prom':>11}")
    print("  " + "-" * 58)
    for name, d in sorted(stats["by_person"].items()):
        print(
            f"  {name:<25} {d['total']:>9} {d['approved']:>10} "
            f"{d['avg_confidence']:>10.1f}"
        )
    print("=" * 60 + "\n")
    print(f"  Archivo de log: {LOG_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_summary()
