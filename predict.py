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
    data = pd.read_csv("data/p_properties.csv")
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

    print(cat_features)
    print(cat_features1)
    #imputer = artifacts["imputer"]
    preprocessor = artifacts["preprocessor"]
    preprocessor1 = artifacts1["preprocessor"]
    #enc = artifacts["enc"]
    model = artifacts["model"]
    model1 = artifacts1["model"]
    # Extract the used data
    data_bxl = data[num_features + fl_features + cat_features]
    data_rest = data[num_features1 + fl_features1 + cat_features1]
    data_bxl = data_bxl[data_bxl["region"].isin(["Brussels-Capital"])]
    data_rest = data_rest[data_rest["region"]!="Brussels-Capital"]

    #data_rest = data_rest[~data_rest["region"].isin(["Brussels-Capital"])]

    print(data_bxl)
    #data_rest = data[~data["region"].isin(["Brussels-Capital"])]
    print(data_rest)

    #data_rest = data[num_features + fl_features + cat_features ]
    # Apply imputer and encoder on data

    #data_bxl[num_features] = imputer.transform(data_bxl[num_features])
    data_bxl = preprocessor.transform(data_bxl) 
    data_rest = preprocessor1.transform(data_rest)
    print(data_bxl.shape)
    print(data_rest.shape)
    print(data_rest)
    numeric_transformer = preprocessor.named_transformers_['pipeline-1']
    categorical_transformer = preprocessor.named_transformers_['pipeline-2']
    numeric_feature_names = numeric_transformer.named_steps['standardscaler'].get_feature_names_out(num_features)
    categorical_feature_names = categorical_transformer.named_steps['onehotencoder'].get_feature_names_out(cat_features)
    numeric_transformer1 = preprocessor1.named_transformers_['pipeline-1']
    categorical_transformer1 = preprocessor1.named_transformers_['pipeline-2']
    numeric_feature_names1 = numeric_transformer1.named_steps['standardscaler'].get_feature_names_out(num_features)
    categorical_feature_names1 = categorical_transformer1.named_steps['onehotencoder'].get_feature_names_out(cat_features)
    columns_bxl = np.concatenate([numeric_feature_names, categorical_feature_names, fl_features])
    columns_rest = np.concatenate([numeric_feature_names1, categorical_feature_names1, fl_features1])
    data_bxl = pd.DataFrame(data_bxl, columns=columns_bxl)
    data_rest = pd.DataFrame(data_rest, columns=columns_rest)


    #data_cat = enc.transform(data[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    #data = pd.concat(
        #   [
        #      data[num_features + fl_features].reset_index(drop=True),
        #],
        #axis=1,
    #)
    # Make predictions
    #bxl_data = data_bxl[data_bxl["region"].isin(["Brussels-Capital"])]
    #rest_data = data_rest[~data_rest["region"].isin(["Brussels-Capital"])]

    predictions_bxl = model.predict(data_bxl)
    predictions_rest = model1.predict(data_rest)
    predictions= np.concatenate([predictions_bxl, predictions_rest])

    

    

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
