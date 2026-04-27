FROM python:3.9-slim

WORKDIR /app

# Copier les fichiers
COPY heart_nn_model.pkl .
COPY scaler.pkl .
COPY app.py .

# Installer les dépendances directement
RUN pip install fastapi uvicorn joblib numpy scikit-learn pandas

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
