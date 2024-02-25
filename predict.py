import click
import joblib
import pandas as pd
import numpy as np

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
    data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    artifacts = joblib.load("models/artifacts_all.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Extract the used data
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

    data = data[num_features + fl_features + cat_features]

    # Apply imputer and encoder on data
    data[num_features] = imputer.transform(data[num_features])
    data_cat = enc.transform(data[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            data[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions
    predictions = model.predict(data)
    

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

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
