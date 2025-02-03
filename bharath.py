import pandas as pd



# Load dataset

url = "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv"

df = pd.read_csv(url)



# Display first few rows

print(df.head())

# Handling missing values

df.fillna(df.median(), inplace=True)



# Selecting relevant features

selected_features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]

X = df[selected_features]

y = df["SalePrice"]



# Splitting data into train and test sets

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Train the model

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



# Evaluate the model

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")

import joblib



joblib.dump(model, "house_price_model.pkl")

from fastapi import FastAPI

import joblib

import pandas as pd



app = FastAPI()



# Load the trained model

model = joblib.load("house_price_model.pkl")



@app.get("/")

def home():

    return {"message": "Welcome to the House Price Prediction API"}



@app.post("/predict/")

def predict(features: dict):

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    return {"predicted_price": prediction}import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv"
df = pd.read_csv(url)

# Display first few rows
print(df.head())
# Handling missing values
df.fillna(df.median(), inplace=True)

# Selecting relevant features
selected_features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = df[selected_features]
y = df["SalePrice"]

# Splitting data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
import joblib

joblib.dump(model, "house_price_model.pkl")
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict/")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"predicted_price": prediction}
