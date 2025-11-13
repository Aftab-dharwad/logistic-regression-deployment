# Titanic Survival Prediction – Logistic Regression Deployment

This project is a simple end-to-end implementation of Logistic Regression, where I trained a model on the Titanic dataset to predict a passenger’s survival probability. After training the model, I deployed it using **Streamlit**, so the prediction can be tested using an interactive web interface.

The main idea of the project is to understand how a machine learning model is created, saved, and finally deployed for real-time usage.

---

## 🚢 Project Description

The model takes basic passenger information such as:

- Ticket Class  
- Gender  
- Age  
- Family members travelling  
- Fare  
- Embarkation point  

and predicts the chances of survival.

I used Logistic Regression because it is simple, interpretable, and works well for binary classification problems.

---

## 📂 Files in This Repository

| File | Description |
|------|-------------|
| **app.py** | Streamlit app used to deploy the logistic regression model |
| **model.pkl** | Trained logistic regression model saved using pickle |
| **Logistic_Regression_Model.ipynb** | Notebook containing training, preprocessing, model building, and evaluation |
| **requirements.txt** | All Python dependencies required to run the app locally |
| **screenshots/** | UI output screenshots of predictions |

---

## 🖥️ How to Run the Project (Windows)

Follow these steps to run the project locally:

### 1️⃣ Create a virtual environment  
```powershell
python -m venv venv

2️⃣ Activate the virtual environment
powershell
.\venv\Scripts\activate


3️⃣ Install all required dependencies
powershell
pip install -r requirements.txt


4️⃣ Run the Streamlit application
powershell
streamlit run app.py


After running the app, your default browser will open with the Titanic Survival Predictor interface.
You can enter passenger details and receive the survival probability along with feature impact visualization.