# Predictor de Calificación Final (G3)

Aplicación web para predecir la calificación final de estudiantes de educación secundaria usando un modelo **Extra Trees Regressor** entrenado con la metodología CRISP-DM.

## Dataset
**Student Performance in Secondary Education**  
Variable dependiente: `G3` (calificación final, escala 0–20)

## Características usadas por el modelo
| Variable | Descripción |
|---|---|
| G1 | Calificación del primer periodo (0–20) |
| G2 | Calificación del segundo periodo (0–20) |
| failures | Número de fallas académicas previas (0–4) |
| studytime | Tiempo de estudio semanal (1=<2h, 2=2–5h, 3=5–10h, 4=>10h) |
| absences | Número de ausencias escolares (0–93) |
| higher | Desea cursar educación superior (yes/no) |

## Instalación local
```bash
pip install -r requirements.txt
python app.py
```
Abrir http://localhost:5000

## Despliegue en Render
1. Sube el repositorio a GitHub.
2. Crea un nuevo **Web Service** en Render apuntando al repo.
3. Render detecta `render.yaml` automáticamente.

## Alumna
**Zenaida González Ramírez** · Matrícula 20231176 · Grupo 8B
