import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/p_properties.csv")

    data = data[data["region"]!="Brussels-Capital"]
        # Define features to use
    num_features = ["nbr_frontages", 'nbr_bedrooms', "latitude", "longitude", "total_area_sqm",
                            'surface_land_sqm','terrace_sqm','garden_sqm',"nb_epc", 'state_building',]
    fl_features = ["fl_terrace", 'fl_garden', 'fl_swimming_pool',"property_type"]
    cat_features = ['heating_type',
                            'locality', 'subproperty_type','region']


    data = data[data["price"] <= 10000000]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=404
    )
        # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    #imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.fit_transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])


    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    

   
    
    
    best_params = {'subsample': 0.8, 'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.1, 'lambda': 1, 'gamma': 5, 'colsample_bytree': 0.6, 'alpha': 0}



    model = XGBRegressor(**best_params)

    model.fit(X_train, y_train)


    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(mean_absolute_error(y_test, model.predict(X_test)))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")
   
    # Save the model
    artifacts1 = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts1, "models/artifacts_rest.joblib")


if __name__ == "__main__":
    train()
    