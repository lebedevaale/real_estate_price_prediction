# Prediction of the US real estate prices with gradient boosting models

Purpose: predict real estate prices for different statistical areas in the US on the weekly macro and finance data six month in advance

Data: weekly data for US statistical areas by [HAUS](https://haus.com/resources/the-common-haus-price-index) from 2010 to 2023 cleared from artefacts in the older years
* Train - 70%
* Validation - 15%
* Test - 15%
* Alternative test - Case-Shiller S&P index (SPCS20RSA) from 2004 to 2024

Models (tuned with Optuna):
* LightGBM
* XGBoost
* Catboost

Predictions are stacked on the validation data via linear regression with constant and then checked with both testing sets (original one is primary in this case).

Variables:
* Federal Funds Effective Rate (DFF)
* CPI (CPIAUCSL)
* VIX (VIXCLS) 
* PPI (PCU44414441) 
* MortgageRate30 (MORTGAGE30US) 
* Electricity (CUSR0000SEHF01)
* Water (CUSR0000SEHG)
* Plywood (WPU083)
* Steel (WPU101)
* Glass (PCU3272132721)
* Concrete (PCU32733273)
* Unemployment (UNRATE)
* Yield10Y (DGS10)
* DJI (^DJI)
* S&P500 (^GSPC)

All except for the last two variables are provided by FRED, the last two are provided by the Yahoo Finance. In both cases data was retrieved via official API with the help of python libraries (pyfredapi and yfinance accordingly).

During the feature generation all of these variables (plus the target one) are used to calculate logreturns over the previous 4, 13 and 52 weeks (1, 3 and 6 months). These logreturns are added to the dataset in order to capture some momentum of the market.