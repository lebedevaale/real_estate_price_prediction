import time
import shap
import optuna
import operator
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import statsmodels.api as sm
import plotly.graph_objects as go
from contextlib import contextmanager
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error as mse

#---------------------------------------------------------------------------------------------------------------------------------------

def variables_dynamics(data,
                       groupby:str):

    """
    Function for the plotting of the dynamics for the variables

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    groupby : str
        Column to groupby
    mean_only : bool = False
        Whether to plot only means and not min-max

    Prints:
    --------------------
    Dynamics of the variables
    """

    # Creating grid of subplots
    fig = make_subplots(rows = len(data.columns), cols = 1, subplot_titles = data.columns)

    # Scattering returns
    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(x = data.index, y = data[col], mode = 'lines', name = col, row = i + 1, col = 1))

    # Update layout
    fig.update_layout(
        showlegend = False,
        template = 'plotly_dark',
        font = dict(size = 20),
        height = 300 * len(data.columns),
        width = 1200
    )

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------------------------------------------------------

def heatmap(data):

    """
    Function for the plotting of the correlation heatmap

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    
    Prints:
    --------------------
    Correlation heatmap
    """

    # Creating grid of subplots
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ["Pearson Correlation", "Spearman Correlation"])

    # Add trace for each correlation matrix
    z1 = data.corr(method = 'pearson')
    z2 = data.corr(method = 'spearman')
    z = [z1, z2]
    for i in range(len(z)):
        fig.add_trace(go.Heatmap(z = z[i][::-1],
                                 x = data.columns,
                                 y = data.columns[::-1],
                                 text = z[i][::-1].round(2),
                                 texttemplate = "%{text}",
                                 zmin = -1, zmax = 1), 
                                 row = 1, col = i + 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        template = 'plotly_dark',
        font = dict(size = 14),
        height = 600,
        width = 1600
    )
    fig.update_annotations(font_size = 30)

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------------------------------------------------------

def remove_most_insignificant(X_val, X_test, results):
    """
    Function for the removal of the most insignificant variables from the model

    Parameters
    ----------
    X_val : DataFrame
        Set of X for the validation of the model
    X_test : DataFrame
        Set of X for the testing of the model
    results : model
        Fitted statsmodels model

    Returns
    ----------
    X_val : DataFrame
        Optimized set of X for the validation of the model
    X_test : DataFrame
        Optimized set of X for the testing of the model
    """
    # use operator to find the key which belongs to the maximum value in the dictionary:
    max_p_value = max(results.pvalues.iteritems(), key = operator.itemgetter(1))[0]
    # this is the feature you want to drop:
    X_val.drop(columns = max_p_value, inplace = True)
    X_test.drop(columns = max_p_value, inplace = True)

    return X_val, X_test

#---------------------------------------------------------------------------------------------------------------------------------------

def OLS_optimization(Y_val, X_val, Y_test, X_test, p_value_bord:float = 0.05, log:bool = False):
    """
    Function for the optimization of OLS

    Parameters
    ----------
    Y_val, Y_test : array
        Target variable for the regression
    X_val, X_test : DataFrame
        Set of X for the model
    p_value : float = 0.05
        Maximum acceptable p-value for the coefficient

    Returns
    ----------
    results : model
        Fitted statsmodels model
    val_rmse : float
        RMSE score for the validation
    test_rmse : float
        RMSE score for the test
    Y_val_pred : array
        Prediction for the validation
    Y_test_pred : array
        Prediction for the test
    log : bool = False
        Whether target needs to be raised back to exponent
    """
    
    insignificant_feature = True
    while insignificant_feature:
        model = sm.OLS(Y_val, X_val)
        results = model.fit()
        significant = [p_value < p_value_bord for p_value in results.pvalues]
        if all(significant):
            insignificant_feature = False
        else:
            if X_val.shape[1] == 1:  # if there's only one insignificant variable left
                print('No significant features found')
                results = None
                insignificant_feature = False
            else:
                X_val, X_test = remove_most_insignificant(X_val, X_test,  results)
    print(results.summary())

    Y_val_pred = results.predict(X_val)
    Y_test_pred = results.predict(X_test)
    if log == False:
        val_rmse = mse(Y_val_pred, Y_val, squared = False)
        test_rmse = mse(Y_test_pred, Y_test, squared = False)
    else:
        val_rmse = mse(np.exp(Y_val_pred), np.exp(Y_val), squared = False)
        test_rmse = mse(np.exp(Y_test_pred), np.exp(Y_test), squared = False)
    print('Validation score for the stacked model is: ', round(val_rmse, 6))
    print('Test score for the stacked model is: ', round(test_rmse, 6))

    return results, val_rmse, test_rmse, Y_val_pred, Y_test_pred