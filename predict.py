import click
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
def predict(input_dataset, output_dataset):

    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    data = pd.read_csv("badboy.csv")
    data["original_index"] = data.index
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    artifacts = joblib.load("models/artifacts.joblib")
    artifacts1 = joblib.load("models/artifacts1.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    num_features1 = artifacts1["features"]["num_features"]
    fl_features1 = artifacts1["features"]["fl_features"]
    cat_features1 = artifacts1["features"]["cat_features"]


    #imputer = artifacts["imputer"]
    preprocessor = artifacts["preprocessor"]
    preprocessor1 = artifacts1["preprocessor"]
    #enc = artifacts["enc"]
    model = artifacts["model"]
    model1 = artifacts1["model"]

    energy_class_bxl =  {
    'A++': (-20, 0),
    'A+': (0, 15),
    'A': (16, 30),
    'A-': (31, 45),
    'B+': (46, 62),
    'B': (63, 78),
    'B-': (79, 95),
    'C+': (96, 113),
    'C': (114, 132),
    'C-': (133, 150),
    'D+': (151, 170),
    'D': (171, 190),
    'D-': (191, 210),
    'E+': (211, 232),
    'E': (233, 253),
    'E-': (254, 275),
    'F': (276, 345),
    'G': (346, 800)
    }
    energy_class_fld = {
    'A+': (-20,0),
    'A': (0, 100),
    'B': (100,200),
    'C': (200,300),
    'D': (300, 400),
    'E': (400, 500),
    'F': (500, 900)
    }
    energy_class_wal = {
        'A++': (-20, 0),
        'A+': (0, 45),
        'A': (45, 85),
        'B': (85, 170),
        'C': (170, 255),
        'D': (255, 340),
        'E': (340, 425),
        'F': (425, 510),
        'G': (510, 900)
    }
    state_mapping = {'JUST_RENOVATED': 6, 'AS_NEW': 5, 'GOOD': 4, 'TO_BE_DONE_UP': 3, 'TO_RENOVATE': 2, 'TO_RESTORE': 1}
    property_type={'APARTMENT': 1, 'HOUSE': 0}
    # Apply mappings to create new numerical columns
    def map_to_numerical(column, mapping):
        return column.map(mapping)

    data["state_building"] = map_to_numerical(data["state_building"], state_mapping)
    data["property_type"] = map_to_numerical(data["property_type"], property_type)
    def random_value_for_energy_class(row):
        primary_energy_column = row.get('primary_energy_consumption_sqm')
        epc_column = row.get('epc')
        region = row.get('region')
        
        if pd.isna(primary_energy_column) and epc_column == 'MISSING':
            return np.nan
        elif pd.notna(primary_energy_column):
            return primary_energy_column
        elif region == 'Brussels-Capital':
            lower_bound, upper_bound = energy_class_bxl.get(epc_column, (0, 0))
            return np.random.uniform(lower_bound, upper_bound)
        elif row['region'] == 'Wallonia':
            lower_bound, upper_bound = energy_class_wal.get(row['epc'], (0, 0))
            return np.random.uniform(lower_bound, upper_bound)
        elif row['region'] == 'Flanders':
            lower_bound, upper_bound = energy_class_fld.get(row['epc'], (0, 0))
            return np.random.uniform(lower_bound, upper_bound)
        else:
            return np.nan
    data['nb_epc'] = data.apply(random_value_for_energy_class, axis=1)
    # data_rest['nb_epc'] = data_rest.apply(random_value_for_energy_class, axis=1)
    # Extract the used data
    index_col = ["original_index"]

    data_bxl = data[num_features + fl_features + cat_features + index_col]
    data_rest = data[num_features1 + fl_features1 + cat_features1 + index_col]
    #apply pre-processing to both

    # Apply the function to create a new column 'nb_epc'


    # Display the DataFrame with the new column


    data_bxl = data_bxl[data_bxl["region"].isin(["Brussels-Capital"])]
    data_rest = data_rest[data_rest["region"]!="Brussels-Capital"]

    index_bxl = data_bxl["original_index"]
    index_rest = data_rest["original_index"]

    # Apply imputer and encoder on data

    data_bxl = preprocessor.transform(data_bxl) 
    data_rest = preprocessor1.transform(data_rest)


    numeric_transformer = preprocessor.named_transformers_['pipeline-1']
    categorical_transformer = preprocessor.named_transformers_['pipeline-2']
    numeric_feature_names = numeric_transformer.named_steps['standardscaler'].get_feature_names_out(num_features)
    categorical_feature_names = categorical_transformer.named_steps['onehotencoder'].get_feature_names_out(cat_features)
    #numeric_transformer1 = preprocessor1.named_transformers_['pipeline-1']
    #categorical_transformer1 = preprocessor1.named_transformers_['pipeline-2']
    #numeric_feature_names1 = numeric_transformer1.named_steps['standardscaler'].get_feature_names_out(num_features)
    #categorical_feature_names1 = categorical_transformer1.named_steps['onehotencoder'].get_feature_names_out(cat_features)
    columns_bxl = np.concatenate([numeric_feature_names, categorical_feature_names, fl_features])
    #columns_rest = np.concatenate([numeric_feature_names1, categorical_feature_names1, fl_features1])
    columns_rest = preprocessor1.get_feature_names_out()
    data_bxl = pd.DataFrame(data_bxl, columns=columns_bxl)
    data_rest = pd.DataFrame(data_rest, columns=columns_rest)


    predictions_bxl = model.predict(data_bxl)
    predictions_rest = model1.predict(data_rest)


    predictions_bxl = pd.DataFrame(predictions_bxl)
    predictions_rest = pd.DataFrame(predictions_rest)

    predictions_bxl = predictions_bxl.set_index(index_bxl)
    predictions_rest = predictions_rest.set_index(index_rest)


    predictions = pd.concat([predictions_bxl, predictions_rest], ignore_index=False)
    predictions = predictions.sort_index()

    

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions[0]}).to_csv(output_dataset, index=False)


    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
    predict()