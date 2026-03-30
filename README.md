# 🏠 House Price Prediction

## 📌 Overview

This project predicts house prices based on features like size (BHK), total square feet, number of bathrooms, balconies, and location.

## 📊 Dataset

* Bengaluru House Data
* Includes features like:

  * size
  * total_sqft
  * bath
  * balcony
  * location
  * price

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pickle (model saving)

## ⚙️ Features

* Data cleaning and preprocessing
* Handling missing values
* Outlier detection and removal
* Feature engineering
* One-hot encoding for categorical data
* Linear Regression model

## 🧠 Model

* Algorithm: Linear Regression
* Evaluation:

  * MAE (Mean Absolute Error)
  * R² Score

## ▶️ How to Run

### 1. Train Model

```bash
python train.py
```

### 2. Predict Price

```bash
python predict.py
```

Then enter:

* BHK
* Total Sqft
* Bathrooms
* Balcony
* Location

## 📈 Example

Input:

* Size: 2
* Total Sqft: 1200
* Bath: 2
* Balcony: 1
* Location: Whitefield

Output:
Predicted Price: XX lakh

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── model.pkl
├── columns.json
├── train.py
├── predict.py
├── README.md
```

## 🚀 Future Improvements

* Add a simple web interface (using Streamlit or Flask)
* Try more machine learning models like Random Forest
* Improve prediction accuracy
* Deploy the project online

