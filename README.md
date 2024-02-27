# Prediction of the US real estate prices with gradient boosting models

Data: weekly data for US statistical areas by [HAUS](https://haus.com/resources/the-common-haus-price-index) from 2010 to 2023 cleared from artefacts in the older years
* Train - 70%
* Validation - 15%
* Test - 15%
* Alternative test - Case-Shiller S&P index from 2004 to 2024

Models:
* LightGBM
* XGBoost
* Catboost

Predictions are stacked on the validation data via linear regression with constant and then checked with both testing sets (original one is primary in this case)