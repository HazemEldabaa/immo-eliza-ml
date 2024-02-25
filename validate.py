import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
predictions = pd.read_csv("output/predictions.csv")
actual = pd.read_csv("data/input.csv")


test_score = r2_score(actual['price'], predictions)
print(f"Mean Absolute Error: {mean_absolute_error(actual["price"], predictions)}")
print(f"R² score: {test_score}")
