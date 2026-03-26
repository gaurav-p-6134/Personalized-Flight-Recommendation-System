# ✈️ Personalized Flight Recommendation System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://flight-recommendation-api.onrender.com)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://personalized-flight-recommendation.vercel.app)
[![ML](https://img.shields.io/badge/Model-XGBoost%20Ranking-EE4C2C?style=for-the-badge&logo=xgboost&logoColor=white)]()

A full-stack, end-to-end machine learning application designed to optimize flight selection for travelers. This system utilizes a **Pairwise Ranking (Learning to Rank)** approach to balance cost, travel duration, and airline quality based on real-world airline reviews and flight metadata.

## 🚀 Live Links
* **Web Dashboard:** [https://personalized-flight-recommendation.vercel.app](https://personalized-flight-recommendation.vercel.app)
* **Interactive API Docs:** [https://flight-recommendation-api.onrender.com/docs](https://flight-recommendation-api.onrender.com/docs)

---

## 🛠️ Tech Stack
* **Machine Learning:** XGBoost (Pairwise Ranking), Scikit-learn, Optuna.
* **Data Engineering:** Polars & DuckDB (High-performance data processing).
* **Backend:** FastAPI, Uvicorn, Pydantic (RESTful API).
* **Frontend:** React.js, Vite, Tailwind CSS v4.
* **Cloud & MLOps:** Render (API Hosting), Vercel (Frontend), Model Compression (UBJ).

---

## 📈 Key Results & Performance
The model was trained and validated using the **Aeroclub RecSys 2025** dataset.
* **Model Performance:** Achieved a **HitRate@3 of 0.49068**, meaning the system correctly predicts a relevant flight in the top 3 results for ~49% of users.
* **Latency:** Optimized inference pipeline using binary model formats to ensure sub-100ms response times.
* **Robustness:** Validated across 1,000+ simulated user query scenarios to ensure stable recommendations.

---

## 🧠 System Architecture
1.  **Preprocessing:** Live JSON data is transformed using **Polars** to handle high-cardinality categorical features (Airlines, Routes) and numeric scaling.
2.  **Ranking Logic:** Uses an **XGBRanker** to learn user preference patterns, prioritizing "Best Value" (balancing Price and Time efficiency).
3.  **API Layer:** A containerized FastAPI server that handles model loading and real-time inference.
4.  **UI Layer:** A modern React dashboard that reorders flight results in real-time based on AI confidence scores.

---

## 📂 Project Structure
```text
Personalized-Flight-Recommendation-System/
├── flight-backend/          # Python FastAPI Server
│   ├── assets/              # Compressed Model & Preprocessing Maps
│   ├── main.py              # API Inference Logic
│   └── requirements.txt     # Backend Dependencies
├── flight-frontend/         # React + Vite Dashboard
│   ├── src/                 # UI Components & Search Logic
│   └── package.json
├── flight-recommendation-system/notebook/               # Jupyter Notebooks (Model Training & EDA)
└── README.md
