import joblib
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/p_properties.csv")

    data = data[data["region"]!="Brussels-Capital"]
        # Define features to use
    num_features = ["nbr_bedrooms", "nbr_frontages","nb_epc","state_building", "total_area_sqm","surface_land_sqm"]
    fl_features = ["fl_terrace","fl_floodzone","fl_swimming_pool","fl_garden","property_type"]
    cat_features = ["heating_type"]
    data = data[data["price"] <= 1000000]
    #data = data[~data["heating_type"].isin(["WOOD","PELLET"])]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=404
    )

     # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])
   



    model = HistGradientBoostingRegressor(categorical_features=[11])

    model.fit(X_train, y_train)


    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(mean_absolute_error(y_test, model.predict(X_test)))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")
   
    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        #"enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts1.joblib")


if __name__ == "__main__":
    train()
    