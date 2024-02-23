# Model card

## Project context

The Immo-Eliza machine learning models were created to predict real estate prices of any property in Belgium based on distinct features. This project is a follow up of  the [Immo-Eliza scraping project](https://github.com/sahar-mahmoudi/immo-eliza-goats), where about 80,000 properties were scraped, cleaned and analysed from Immo-Web
## Data

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
## Model details

The models tested 
|Brussels|Flanders & Wallonia||
|--|--|--|
|XGBoost|XGBoost||
|<table> <tr><th>R-Score</th><tr><td>Train R² score: 0.9530388098450423
Test R² score: 0.7856024852551156</tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.832349411827084
Test R² score: 0.7470771575728206</td></tr> </table>|
|HistGradientBoostRegressor|HistGradientBoostRegressor||
|<table> <tr><th>R-Score</th><tr><td>Train R² score: 0.8952725299266017
Test R² score: 0.7884815695408979</td></tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.7657218325474691
Test R² score: 0.7309478342981872</td></tr> </table>|
|RandomForestRegressor|RandomForestRegressor||
|<table> <tr><th>R-Score</th></tr><tr><td>Train R² score: 0.9709258962769078
Test R² score: 0.7935725791558854</td></tr> </table>| <table> <tr><th>R2-Score</th></tr><tr><td>Train R² score: 0.9622498184528874
Test R² score: 0.7390053232454324</td></tr> </table>|

## Visuals

### Kernel Density Estimate:
#### Brussels Model:
![kdebxl](https://i.ibb.co/VmZRgfm/image.png)
#### Flanders & Wallonia Model:
![kderest](https://i.ibb.co/NpHd1pk/image.png)

## Limitations

It's overfitting like crazy

## Usage
1. To preprocess the input data 

```bash
python preprocessing.py
```
This will create a "p_properties.csv" in your /data directory

2. To train the two models on pre-processed data:

```bash
python train.py
python train1.py
```

3. To generate predictions in /output (replace "data/input.csv" with your input)

```bash
python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
```


## Maintainers

Who to contact in case of questions or issues?

God