# Model card

## 🏡Project context

The Immo-Eliza machine learning models were created to predict real estate prices of any property in Belgium based on distinct features. This project is a follow up of  the [collaborative Immo-Eliza scraping project](https://github.com/sahar-mahmoudi/immo-eliza-goats), where about 80,000 properties were scraped, cleaned and analysed from Immo-Web
## 📊Data

The input dataset was the cleaned output of the Immo-Scraper of approximately 76,000 properties
|Selected Features| Target| |
|---|---|---|
Locality|Price|
Number of Frontages||
Property Type||
Bedroom Count||
Total Area (m²)||
Equipped Kitchen||
Flood Zone Type||
Surface Land||
Swimming Pool||
State of Building||
Primary Energy Consumption per m²||
Heating Type||
Latitude||
Longitude||
## 🔍Model details

Two model architectures were created and compared for this project:
- One general model for all regions
- Two models, one for Brussels, the other for Flanders & Wallonia

The reason for this comparison was due to the data analysis stage where the Brussels-Capital region was deemed to have different patterns compared to the rest of the regions.

The models tested for the two model architecture:

|Brussels|Flanders & Wallonia||
|--|--|--|
|**XGBoost**|**XGBoost**||
|<table> <tr><th>R-Score</th><tr><td>Train R² score: 0.9530388098450423
Test R² score: 0.7856024852551156</tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.832349411827084
Test R² score: 0.7470771575728206</td></tr> </table>|
|**HistGradientBoostRegressor**|**HistGradientBoostRegressor**||
|<table> <tr><th>R-Score</th><tr><td>Train R² score: 0.8952725299266017
Test R² score: 0.7884815695408979</td></tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.7657218325474691
Test R² score: 0.7309478342981872</td></tr> </table>|
|**RandomForestRegressor**|**RandomForestRegressor**||
|<table> <tr><th>R-Score</th></tr><tr><td>Train R² score: 0.9709258962769078
Test R² score: 0.7935725791558854</td></tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.9622498184528874
Test R² score: 0.7390053232454324</td></tr> </table>|


|Model for all regions|
|--|
|**XGBoost**|
|Train R² score: 0.8857624394661134
Test R² score: 0.7651603502422053|

## ✔️Validation
A secret data-set was hidden from the model during the training and testing process, to evaluate how the models preform on unseen data.

|Two-Models|Model for all regions|
|--|--|
|**XGBoost**|**XGBoost**|
|Mean Absolute Error: 91566.93684826371|Mean Absolute Error: 93854.67368142928
|R² score: 0.6614795325265299|R² score: 0.761807695250265|
|||

## 🎨Visuals

### Kernel Density Estimate (Blue=Actual, Orange=Predicted):
#### Brussels Model:
![kdebxl](https://i.ibb.co/VmZRgfm/image.png)
#### Flanders & Wallonia Model:
![kderest](https://i.ibb.co/NpHd1pk/image.png)

## 🚫Limitations

Your ```input.csv``` file must match the format of the sample provided to make appropriate predictions

## 👩‍💻Usage
- To preprocess the input data 

```bash
python preprocessing.py
```
This will create a "p_properties.csv" in your /data directory

- To train the two models on pre-processed data:

```bash
python train_bxl.py
python train_rest.py
```
- To train the general model on pre-processed data:
```bash
python train_all.py
```
- To generate predictions for the two-model in /output (You can replace "data/input.csv" with another input of your choice)

```bash
python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
```
- To generate predictions for the general model
```bash
python .\predict_all.py -i "data\input.csv" -o "output\predictions.csv"
```

- To validate the models on the secret data, make sure that ```predictions.csv``` has the predictions of the model intended for validation, then run:

```bash
python validate.py
```

## 🎉Conclusion
Despite earlier hypothesis of regions needing seperate models to be able to capture their respective patterns, my comparison of the two approaches shows that XGBoost is able to capture these complex relationship patterns more effectively within one model in this case.
## 👨‍💼Maintainers

Me