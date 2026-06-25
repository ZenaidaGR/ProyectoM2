import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import traceback


BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"
REQUIRED_MODULES = ("flask", "joblib", "numpy", "pandas", "sklearn")


def _has_required_modules():
    return all(importlib.util.find_spec(module) for module in REQUIRED_MODULES)


def _same_file(left, right):
    try:
        return Path(left).resolve().samefile(Path(right).resolve())
    except OSError:
        return Path(left).resolve() == Path(right).resolve()


def _create_local_venv():
    command = ["py", "-3.11", "-m", "venv", str(VENV_DIR)]
    try:
        subprocess.check_call(command)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])


def _bootstrap_local_venv():
    if _has_required_modules() or os.name != "nt":
        return

    if not _same_file(sys.executable, VENV_PYTHON):
        if not VENV_PYTHON.exists():
            print("Creando entorno local en ./venv con Python 3.11...", flush=True)
            _create_local_venv()

        print("Reiniciando con el Python del entorno local...", flush=True)
        raise SystemExit(subprocess.call([str(VENV_PYTHON), *sys.argv]))

    if REQUIREMENTS_PATH.exists():
        print("Instalando dependencias desde requirements.txt...", flush=True)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_PATH),
        ])
        importlib.invalidate_caches()

    if not _has_required_modules():
        raise RuntimeError(
            "No se pudieron cargar las dependencias. "
            "Revisa la instalacion de requirements.txt."
        )


_bootstrap_local_venv()

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


app = Flask(__name__)

MODEL_PATH = BASE_DIR / "modelo_student_g3.pkl"
pipeline = joblib.load(MODEL_PATH)


def _iter_estimators(estimator):
    yield estimator

    for _, step in getattr(estimator, "steps", []):
        yield from _iter_estimators(step)

    for _, transformer, _ in getattr(estimator, "transformers_", []):
        if transformer in ("drop", "passthrough"):
            continue
        yield from _iter_estimators(transformer)


def _repair_loaded_pipeline(estimator):
    repaired = 0
    for step in _iter_estimators(estimator):
        if (
            isinstance(step, SimpleImputer)
            and not hasattr(step, "_fill_dtype")
            and hasattr(step, "_fit_dtype")
        ):
            step._fill_dtype = step._fit_dtype
            repaired += 1

    if repaired:
        print(f"Compatibilidad scikit-learn: {repaired} imputadores reparados.")


_repair_loaded_pipeline(pipeline)

print("Modelo cargado:", type(pipeline))
print("Pasos del pipeline:", pipeline.named_steps)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        print("Datos recibidos:", data)

        input_data = pd.DataFrame([{
            "G1": float(data["G1"]),
            "G2": float(data["G2"]),
            "failures": int(data["failures"]),
            "studytime": int(data["studytime"]),
            "absences": int(data["absences"]),
            "higher": data["higher"],
        }])

        print("DataFrame enviado al modelo:")
        print(input_data)
        print(input_data.dtypes)

        prediction = pipeline.predict(input_data)[0]
        prediction = round(float(np.clip(prediction, 0, 20)), 2)

        print("Prediccion:", prediction)
        return jsonify({"success": True, "prediction": prediction})

    except Exception as exc:
        print("ERROR COMPLETO:")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
