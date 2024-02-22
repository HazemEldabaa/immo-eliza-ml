import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/p_properties.csv")

    data = data[data["region"]=="Brussels-Capital"]
        # Define features to use
    num_features = ["nbr_bedrooms", "nbr_frontages","nb_epc","state_building", "total_area_sqm","surface_land_sqm","longitude","latitude"]
    fl_features = ["fl_floodzone","fl_swimming_pool","property_type"]
    cat_features = ["heating_type", "equipped_kitchen","region"]
    data = data[data["price"] <= 1000000]
    #data = data[~data["heating_type"].isin(["WOOD","PELLET"])]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=404
    )

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), num_features),
    (make_pipeline(OneHotEncoder(handle_unknown='ignore')), cat_features),
    remainder='passthrough'
    )
    

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(X_train)

    # Train the model
    model = RandomForestRegressor(n_jobs=-1)
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
        "preprocessor":preprocessor,
        #"imputer": imputer,
        #"enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()
    