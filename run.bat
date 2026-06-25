@echo off
if not exist venv (
    py -3.10 -m venv venv
    call venv\Scripts\activate
    pip install flask==3.0.3 joblib==1.4.2 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.7.1
) else (
    call venv\Scripts\activate
)
python app.py
 