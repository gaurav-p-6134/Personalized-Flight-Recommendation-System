# ✈️ Personalized Flight Recommendation System

## 🚀 Project Overview
This project is a sophisticated recommendation system designed to provide business travelers with tailored flight options.It processes over 15,000 real-world airline reviews from SKYTRAX along with flight metadata to understand user preferences.The core of the system is a **collaborative filtering algorithm** optimized to balance key business traveler preferences: cost, travel time, and overall airline quality.

## 🛠️ Tech Stack
- **Languages:** Python
- **Data Processing:** DuckDB, Polars
- **Machine Learning:** Scikit-learn, LightGBM
- **Hyperparameter Tuning:** Optuna

## 📈 Key Results & Performance
The model's effectiveness was rigorously tested to ensure high-quality recommendations.
* **Model Performance:** Achieved a **HitRate@3 of 0.49068**, meaning the system correctly placed a relevant flight in the top 3 recommendations for nearly 49% of simulated users.
* **Robustness:** The model's stability was validated against **1,000 simulated user query scenarios** to ensure reliable performance under various conditions.


## 📂 Project Structure
A clean and organized folder structure makes your project easy to navigate.
flight-recommendation-system
- notebooks (Jupyter notebooks for exploration and analysis)
- src        (Source code (e.g., Python scripts for data processing, model training))
- README.md   (You are here!)
- requirements.txt   (List of Python libraries needed to run the project)
## 💾 Data
The dataset for this project is from the **Aeroclub RecSys 2025 Kaggle Competition**. Due to its large size (5GB), it is not included in this repository.

**To run this project, you must download the data from the competition page:**

1.  **Download from Kaggle:** Visit the [Aeroclub RecSys 2025 Competition Page](https://www.kaggle.com/competitions/aeroclub-recsys-2025) and download the `train.parquet` and `test.parquet` files.
2.  **Place the Data:** Create a folder named `data` in the root of this project folder. Place the downloaded `.parquet` files inside this `data` folder.
3.  **Final Structure:** Your project folder should look like this:

    ```
    /flight-recommendation-system
    |
    ├── data/
    │   ├── train.parquet
    │   └── test.parquet
    ├── notebooks/
    ├── src/
    ├── README.md
    └── requirements.txt
    ```

## 🔧 How to Run
To get this project running locally, follow these steps:
1.  Clone the repository: `git clone https://github.com/your-username/flight-recommendation-system.git`
2.  Navigate to the project directory: `cd flight-recommendation-system`
3.  Install the required dependencies: `pip install -r requirements.txt`
4.  Run the main analysis notebook located in the `/notebooks` folder.
