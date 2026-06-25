from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_student_g3.pkl")
pipeline = joblib.load(MODEL_PATH)

print("Modelo cargado:", type(pipeline))
print("Pasos del pipeline:", pipeline.named_steps)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Datos recibidos:", data)

        input_data = pd.DataFrame([{
            "G1":        float(data["G1"]),
            "G2":        float(data["G2"]),
            "failures":  int(data["failures"]),
            "studytime": int(data["studytime"]),
            "absences":  int(data["absences"]),
            "higher":    data["higher"],
        }])

        print("DataFrame enviado al modelo:")
        print(input_data)
        print(input_data.dtypes)

        prediction = pipeline.predict(input_data)[0]
        prediction = round(float(np.clip(prediction, 0, 20)), 2)

        print("Predicción:", prediction)
        return jsonify({"success": True, "prediction": prediction})

    except Exception as e:
        print("ERROR COMPLETO:")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)