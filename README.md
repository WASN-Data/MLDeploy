# 🎵 MLDeploy - Music Genre Classification

## Overview

Application web de classification automatique de genres musicaux utilisant le machine learning. Le système analyse des fichiers audio et prédit parmi **9 genres** : Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Rock.

## Architecture

```
MLDeploy_NoReggae/
├── artifacts/                  # Modèles et artefacts ML
│   ├── model.pkl              # Modèle Gradient Boosting entraîné
│   ├── scaler.pkl             # StandardScaler pour normalisation
│   ├── label_encoder.pkl      # Encodeur des labels
│   ├── feature_names.pkl      # Noms des 58 features
│   └── checkpoints/           # Historique des modèles réentraînés
├── data/
│   ├── features_30_sec.csv    # Dataset GTZAN (features pré-extraites)
│   ├── ref_data.csv           # Données de référence pour drift
│   └── prod_data.csv          # Données de production (feedback)
├── notebooks/
│   └── 01_EDA_and_Model_Training.ipynb  # EDA + entraînement
├── serving/                   # API FastAPI
│   ├── api.py                 # Endpoints: /predict, /feedback, /health
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── webapp/                    # Interface Streamlit
│   ├── app.py                 # UI avec onglets Classify & Drift Monitor
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── reporting/
    └── drift_report.py        # Générateur de rapports Evidently
```

## Fonctionnalités

### 🎯 Classification
- Upload d'un fichier audio (WAV, MP3, FLAC...)
- Extraction automatique de 58 features GTZAN
- Prédiction avec confiance et top-3 genres

### 📊 MLOps
- **Feedback Loop** : Collecte des corrections utilisateur
- **Retraining automatique** : Réentraînement après 10 feedbacks
- **Checkpoints** : Sauvegarde horodatée de chaque modèle (`model_YYYY-MM-DD_HH-MM-SS.pkl`)
- **Drift Monitoring** : Détection du data drift avec Evidently

## Installation & Lancement

### Option 1 : Docker (Recommandé)

```bash

# Lancer l'API (port 8080)
docker compose -f serving/docker-compose.yml up --build -d

# Lancer la webapp (port 8081)
docker compose -f webapp/docker-compose.yml up --build -d
```

### Option 2 : Local

**Terminal 1 - API :**
```bash
cd serving
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8080
```

**Terminal 2 - Webapp :**
```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py --server.port 8081
```

## URLs

| Service | URL |
|---------|-----|
| API Swagger | http://localhost:8080/docs |
| Webapp | http://localhost:8081 |

## Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict` | POST | Classification d'un fichier audio |
| `/feedback` | POST | Soumettre une correction |
| `/retrain` | POST | Forcer le réentraînement |
| `/health` | GET | État de l'API |
| `/model-info` | GET | Infos sur le modèle |

## Stack Technique

- **ML** : scikit-learn (Gradient Boosting), librosa
- **API** : FastAPI, Uvicorn
- **Frontend** : Streamlit
- **Monitoring** : Evidently
- **Infra** : Docker, Docker Compose

## Dataset

**GTZAN Dataset** (modifié - sans reggae) :
- 900 tracks audio (30 secondes chacun)
- 9 genres × 100 tracks
- 58 features audio pré-extraites

### Features extraites
- **MFCCs** (1-20) : Caractéristiques timbrales
- **Spectral** : Centroid, Bandwidth, Rolloff
- **Chroma** : Contenu harmonique
- **RMS** : Énergie sonore
- **Zero Crossing Rate** : Caractéristiques percussives
- **Tempo** : BPM

## Contributeurs

Projet réalisé dans le cadre du cours "Concepts & Technologies IA" - Ynov 2025-2026  
Nolan BERGER  
Awa GUEYE SECK  
Wafah LEMAISSI  
Sophie CAPRON  


