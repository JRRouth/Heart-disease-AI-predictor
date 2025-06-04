---

❤️ Heart Disease AI Predictor

A machine learning web app built with Streamlit that predicts the risk of heart disease based on user input. It uses a tuned Random Forest Classifier trained on a heart disease dataset.

🚀 Live Demo




---

📂 Project Structure

heart-disease-ai-predictor/
│
├── heart_disease.csv             # Dataset
├── heart_disease_model.pkl       # Trained ML model
├── feature_columns.pkl           # List of model input features
├── scaler.pkl                    # Scaler (if used during training)
├── app.py                        # Streamlit app
├── train_model.py                # Script to train and save model
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


---

📊 Features Used

Age

Sex

Chest Pain Type

Resting Blood Pressure

Serum Cholesterol

Fasting Blood Sugar

Resting ECG

Max Heart Rate Achieved

Exercise Induced Angina

Oldpeak

Slope of ST Segment

Number of Major Vessels

Thalassemia


Categorical features are one-hot encoded.


---

🛠️ How to Run Locally

1. Clone the repository

git clone https://github.com/JRRouth/Heart-disease-AI-predictor.git
cd Heart-disease-AI-predictor


2. Install dependencies

pip install -r requirements.txt


3. Run the app

streamlit run app.py

---

✅ Requirements

Python 3.7+

scikit-learn

pandas

numpy

streamlit

joblib

matplotlib

seaborn


(See requirements.txt)


---

📌 Deployment on Streamlit Cloud

1. Push your code to a GitHub repo


2. Go to Streamlit Cloud


3. Create a new app from your repo


4. Set app.py as the main file


5. Add any required secrets or environment variables (if any)


6. Click Deploy




---

🤖 Model Info

Model: XGBoost(tuned with GridSearchCV)

Performance:

Accuracy: ~90%

ROC AUC: ~0.96




---

📬 Contact

For questions or feedback, feel free to reach out via GitHub issues.


---


