import time
import shap
import nolds
import optuna
import operator
import numpy as np
import pandas as pd
import sklearn as sk
from PyEMD import EMD
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import plotly.io as pio
from scipy.fft import fft
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from contextlib import contextmanager
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import mutual_info_regression

pio.templates.default = "plotly_white"

# Receiving hyperparams for sample modifications
from configparser import ConfigParser
config = ConfigParser()

#---------------------------------------------------------------------------------------------------------------------------------------

def variables_dynamics(data,
                       directory = ''):

    """
    Function for the plotting of the dynamics for the variables

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    directory : str = ''
        Directory where data is stored if it isn't CWD

    Plots:
    --------------------
    Plot with dynamics of the variables
    """

    # Creating grid of subplots
    fig = make_subplots(rows = len(data.columns), cols = 1, subplot_titles = data.columns)

    # Scattering returns
    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(x = data.index, y = data[col], mode = 'lines', name = col), row = i + 1, col = 1)

    # Update layout and save plot
    fig.update_layout(
        showlegend = False,
        font = dict(size = 20),
        height = 300 * len(data.columns),
        width = 1600
    )
    # pio.write_image(fig, directory + f"Data_for_models/dynamics.png", scale = 6, width = 1600, height = 300 * len(data.columns))
    pio.write_image(fig, directory + f"Data_for_models/dynamics.svg", scale = 6, width = 1600, height = 300 * len(data.columns))

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------------------------------------------------------

def heatmap(data, 
            directory = ''):

    """
    Function for the plotting of the correlation heatmap

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    directory : str = ''
        Directory where data is stored if it isn't CWD
    
    Plots:
    --------------------
    Correlation heatmaps
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

    # Update layout and save plot
    fig.update_layout(
        showlegend = False,
        font = dict(size = 20),
        height = 1600,
        width = 3200
    )
    fig.update_annotations(font_size = 30)
    pio.write_image(fig, directory + f"Data_for_models/heatmaps.png", scale = 6, width = 2800, height = 1400)
    pio.write_image(fig, directory + f"Data_for_models/heatmaps.svg", scale = 6, width = 3200, height = 1600)

    # Show the plot
    fig.show()

#-------------------------------------------------------------------------------------------------------

def stationarity(data):
    
    """
    Function for the calculation of stationarity of time series

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis

    Prints:
    --------------------
    res : pd.DataFrame
        Dataframe with results of the stationarity test
    """

    # Calculate metrics of stationarity and level of chaos in the data
    res = pd.DataFrame(columns = ['Variable', 'DF statistics', 'DF p-value', 'Lyapunov LE', 'Hurst E'])
    for col in data.columns:
        stat = adfuller(data[col])
        res.loc[len(res)] = [col, stat[0], stat[1], nolds.lyap_r(data[col]), nolds.hurst_rs(data[col])]
    
    return res

#---------------------------------------------------------------------------------------------------------------------------------------

def remove_most_insignificant(X_val, X_test, results):
    
    """
    Function for the removal of the most insignificant variables from the model

    Inputs:
    ----------
    X_val : DataFrame
        Set of X for the validation of the model
    X_test : DataFrame
        Set of X for the testing of the model
    results : model
        Fitted statsmodels model

    Returns:
    ----------
    X_val : DataFrame
        Optimized set of X for the validation of the model
    X_test : DataFrame
        Optimized set of X for the testing of the model
    """

    # Use operator to find the key which belongs to the maximum value in the dictionary:
    max_p_value = max(results.pvalues.iteritems(), key = operator.itemgetter(1))[0]
    
    # Drop the worst feature
    X_val.drop(columns = max_p_value, inplace = True)
    X_test.drop(columns = max_p_value, inplace = True)

    return X_val, X_test

#---------------------------------------------------------------------------------------------------------------------------------------

def OLS_optimization(Y_val, 
                     X_val, 
                     Y_test, 
                     X_test,
                     p_value_bord:float = 0.05, 
                     log:bool = False,
                     silent_results:bool = False,
                     silent_scores:bool = False):
    
    """
    Function for the optimization of OLS

    Inputs:
    ----------
    Y_val, Y_test : array
        Target variable for the regression
    X_val, X_test : DataFrame
        Set of X for the model
    p_value_bord : float = 0.05
        Maximum acceptable p-value for the coefficient
    log : bool = False
        Whether to raise target and predictions data to the exponent before calculating RMSE
    silent_results : bool = False
        Whether to print whole stats of the regression
    silent_scores : bool = False
        Whether to print scores for validation and test

    Returns:
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
    """
    
    # Iterate while model has insignificant features
    insignificant_feature = True
    while insignificant_feature:
        model = sm.OLS(Y_val, X_val)
        results = model.fit()
        significant = [p_value < p_value_bord for p_value in results.pvalues]
        if all(significant):
            insignificant_feature = False
        else:
            # If there's only one insignificant variable left
            if X_val.shape[1] == 1:  
                print('No significant features found')
                results = None
                insignificant_feature = False
            else:
                X_val, X_test = remove_most_insignificant(X_val, X_test, results)
    if silent_results == False:
        print(results.summary())

    # Calculate validation and test predictions and scores for them
    Y_val_pred = results.predict(X_val)
    Y_test_pred = results.predict(X_test)
    if log == False:
        val_rmse = mse(Y_val_pred, Y_val, squared = False)
        test_rmse = mse(Y_test_pred, Y_test, squared = False)
    else:
        val_rmse = mse(np.exp(Y_val_pred), np.exp(Y_val), squared = False)
        test_rmse = mse(np.exp(Y_test_pred), np.exp(Y_test), squared = False)
    if silent_scores == False:
        print('Validation score for the stacked model is: ', round(val_rmse, 3))
        print('Test score for the stacked model is: ', round(test_rmse, 3))

    return results, val_rmse, test_rmse, Y_val_pred, Y_test_pred

#---------------------------------------------------------------------------------------------------------------------------------------

def OLS_benchmark(lag:int,
                  directory:str = '',
                  silent_results:bool = False):
    
    """
    Main function for the estimation OLS benchmark

    Inputs:
    ----------
    lag : int
        Distance of prediction in weeks
    directory : str = ''
        Directory where data is stored if it isn't CWD
    silent_results : bool = False
        Whether to print whole stats of the regression
    """

    # Import val and test seize and log flag from config
    config.read(directory + 'config.cfg')
    val_size = float(config.get('params', 'val_size'))
    test_size = float(config.get('params', 'test_size'))
    log = bool(config.get('params', 'log'))
    random_state = int(config.get('params', 'random_state'))

    # Load dataset for modelling
    data = pd.read_parquet(directory + 'Data_for_models/final_full.parquet').dropna(subset = [f'target_{lag}_week_fut'])
    data_2008 = pd.read_parquet(directory + 'Data_for_models/final_CS.parquet').dropna(subset = [f'target_{lag}_week_fut'])

    # Split dataset on train, validation and test
    Y = data[f'target_{lag}_week_fut']
    X = sm.add_constant(data.drop(columns = data.columns[data.columns.str.contains('_week_fut')]))
    Y_2008 = data_2008[f'target_{lag}_week_fut']
    X_2008 = sm.add_constant(data_2008.drop(columns = data_2008.columns[data_2008.columns.str.contains('_week_fut')]))
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(
        X, Y, test_size = test_size, random_state = random_state)
    X_train, X_val, Y_train, Y_val = sk.model_selection.train_test_split(
        X_train, Y_train, test_size = (1 - test_size) * val_size, random_state = random_state)
    
    # Train OLS to get scores
    print(f'\n OLS benchmark, {lag} lag:')
    results, train_rmse, val_rmse, _, _ = OLS_optimization(Y_train, X_train, Y_val, X_val, log = log, 
                                                           silent_results = silent_results,
                                                           silent_scores = True)
    
    if log == True:
        test_rmse = mse(np.exp(results.predict(X_test[list(results.params.index)])), np.exp(Y_test), squared = False)
        rmse_2008 = mse(np.exp(results.predict(X_2008[list(results.params.index)])), np.exp(Y_2008), squared = False)
    else:
        test_rmse = mse(results.predict(X_test[list(results.params.index)]), Y_test, squared = False)
        rmse_2008 = mse(results.predict(X_2008[list(results.params.index)]), Y_2008, squared = False)
    print(f'Train score for OLS benchmark is: ', round(train_rmse, 3))
    print(f'Validation score for OLS benchmark is: ', round(val_rmse, 3))
    print(f'Test score for OLS benchmark is: ', round(test_rmse, 3))
    print(f'Test score for OLS benchmark on Case-Shiller data is: ', round(rmse_2008, 3))

#---------------------------------------------------------------------------------------------------------------------------------------

@contextmanager
def timer(logger = None, 
          format_str = '{:.3f}[s]', 
          prefix = None, 
          suffix = None):
    
    """
    Function for the calculating time used for calculations

    Inputs:
    ----------
    logger = None
        Whether to log progress or to show only final results
    format_str = '{:.3f}[s]'
        Format in which time used will be demonstrated
    prefix : str = None
        Prefix for string if you need them to be different
    suffix : str = None
        Suffix for string if you need them to be different

    Prints:
    ----------
    Used time
    """

    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time.time()
    yield
    d = time.time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)

class TreeModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.trn_data = None
        self.val_data = None
        self.model = None

    def train(self,
              params: dict,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray,
              location: str,
              train_weight: np.ndarray = None,
              val_weight: np.ndarray = None,
              train_params: dict = {}
              ):
        if self.model_type == "lgb":
            self.trn_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
            self.val_data = lgb.Dataset(X_val, label=y_val, weight=val_weight)
            self.model = lgb.train(params=params,
                                   train_set=self.trn_data,
                                   valid_sets=[self.trn_data, self.val_data],
                                   **train_params)
            self.save = self.model.save_model(location)
        elif self.model_type == "xgb":
            self.trn_data = xgb.DMatrix(X_train, y_train, weight=train_weight)
            self.val_data = xgb.DMatrix(X_val, y_val, weight=val_weight)
            self.model = xgb.train(params=params,
                                   dtrain=self.trn_data,
                                   evals=[(self.trn_data, "train"), (self.val_data, "val")],
                                   **train_params)
            self.save = self.model.save_model(location)
        elif self.model_type == "cat":
            self.trn_data = cat.Pool(X_train, label=y_train, group_id=[0] * len(X_train))
            self.val_data = cat.Pool(X_val, label=y_val, group_id=[0] * len(X_val))
            self.model = cat.CatBoost(params)
            self.model.fit(
                self.trn_data, eval_set=[self.val_data], use_best_model=True, **train_params)
            self.save = self.model.save_model(location)
        else:
            raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        if self.model_type == "lgb":
            return self.model.predict(
                X, num_iteration=self.model.best_iteration)  # type: ignore
        elif self.model_type == "xgb":
            X_DM = xgb.DMatrix(X)
            return self.model.predict(
                X_DM, ntree_limit=self.model.best_ntree_limit)  # type: ignore
        elif self.model_type == "cat":
            return self.model.predict(X)
        else:
            raise NotImplementedError

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").keys())
        elif self.model_type == "cat":
             return self.model.feature_names_
        else:
            raise NotImplementedError

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").values())
        elif self.model_type == "cat":
            return self.model.feature_importances_
        else:
            raise NotImplementedError

#---------------------------------------------------------------------------------------------------------------------------------------

def run_train_and_inference(X_train, X_val, X_test, 
                            Y_train, Y_val, Y_test, 
                            use_model : str, 
                            model_params : dict, 
                            train_params : dict, 
                            location : str, 
                            log:bool = False, 
                            shaps:bool = True):

    """
    Function for the estimation of the models and calculation of the errors in the models

    Inputs:
    ----------
    X_train, X_val, X_test : array
        Set of X for the model
    Y_train, Y_val, Y_test : array
        Target variable for the regression
    use_model : str
        Name of the model: lgb, xgb or cat
    model_params : dict
        Hyperparameters for the model
    train_params : dict
        Hyperparameters for the training
    location : str
        Directory to save model params
    log : bool = False
        Whether to raise target and predictions data to the exponent before calculating RMSE
    shaps : bool = True
        Whether to calculate SHAP values for the model

    Returns:
    ----------
    train_pred, val_pred, test_pred : arrays
        Predictions for all three parts of the dataset
    scores : dict
        RMSE scores for all three parts of the dataset
    shap_values
        Calculated shap_values for the model if shaps == True, else returns None
    """

    # Train model
    model = TreeModel(model_type = use_model)
    with timer(prefix = "Model training "):
        model.train(
            params = model_params, X_train = X_train, y_train = Y_train,
            X_val = X_val, y_val = Y_val, train_params = train_params,
            location = location)

    # Estimate predictions for the train, val and test
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate RMSE for estimations
    if log == False:
        train_score = mse(Y_train, train_pred, squared = False)
        val_score = mse(Y_val, val_pred, squared = False)
        test_score = mse(Y_test, test_pred, squared = False)
    else:
        train_score = mse(np.exp(Y_train), np.exp(train_pred), squared = False)
        val_score = mse(np.exp(Y_val), np.exp(val_pred), squared = False)
        test_score = mse(np.exp(Y_test), np.exp(test_pred), squared = False)
    print(f'Train score for {use_model} is: ', round(train_score, 2))
    print(f'Validation score for {use_model} is: ', round(val_score, 2))
    print(f'Test score for {use_model} is: ', round(test_score, 2))
    scores = {"train": train_score, "val": val_score, "test": test_score}

    # Calculate SHAP values if needed
    if shaps == True:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_val)
    else:
        shap_values = None

    return train_pred, val_pred, test_pred, scores, shap_values

#---------------------------------------------------------------------------------------------------------------------------------------

def optuna_and_boosting(lag:int, 
                        random_state:int, 
                        directory:str = '',
                        shaps:bool = True):
    
    """
    Main function for the estimation of the boosting models

    Inputs:
    ----------
    lag : int
        Distance of prediction in weeks
    random_state : int
        Seed for the RNG
    directory : str = ''
        Directory where data is stored if it isn't CWD
    shaps : bool = True
        Whether to calculate SHAP values for the model
    
    Prints:
    ----------
    Training process and scores of LightGBM, XGBoost and Catboost models
    Stacking results
    Comparison of the models by scores on validation and test

    Files:
    ----------
    Model params for LightGBM, XGBoost, Catboost and stacking models
    Predictions and target values for validation and testing
    """

    def objective_lgb(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log = True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log = True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 7)
        }
        param.update(param_lgb)

        train_param = {
            "num_boost_round": 500,
            "early_stopping_rounds": 100,
            "verbose_eval": False
        }

        dtrain = lgb.Dataset(X_train, label = Y_train)
        dvalid = lgb.Dataset(X_test, label = Y_test)
        gbm = lgb.train(param, dtrain, valid_sets = dvalid, **train_param)
        Y_val_pred = gbm.predict(X_val)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def objective_xgb(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log = True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log = True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log = True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log = True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }
        param.update(param_xgb)

        train_param = {
            "num_boost_round": 500,
            "early_stopping_rounds": 100,
            "verbose_eval": False
        }

        dtrain = xgb.DMatrix(X_train, label = Y_train)
        dvalid = xgb.DMatrix(X_val, label = Y_val)
        gbm = xgb.train(param, dtrain, evals = [(dtrain, 'train') , (dvalid, 'valid')], **train_param)
        Y_val_pred = gbm.predict(dvalid)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def objective_cat(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 7),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log = True)
        }
        param.update(param_cat)

        train_param = {
            "early_stopping_rounds": 100, 
            "verbose_eval": False
        }

        gbm = cat.CatBoostRegressor(**param)
        gbm.fit(X_train, Y_train, eval_set = [(X_val, Y_val)], **train_param)
        Y_val_pred = gbm.predict(X_val)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def optuna_study(model):

        objectives = {
            "lgb": objective_lgb,
            "xgb": objective_xgb,
            "cat": objective_cat
        }

        sampler = optuna.samplers.TPESampler(seed = random_state)
        study = optuna.create_study(direction = "minimize", sampler = sampler)
        study.optimize(objectives[model], n_trials = 50, n_jobs = 1, gc_after_trial = True, show_progress_bar = True)
        trial = study.best_trial

        return trial

    # Import val and test seize and log flag from config
    config.read(directory + 'config.cfg')
    val_size = float(config.get('params', 'val_size'))
    test_size = float(config.get('params', 'test_size'))
    log = bool(config.get('params', 'log'))

    # Load dataset for modelling
    data = pd.read_parquet(directory + 'Data_for_models/final_full.parquet').dropna(subset = [f'target_{lag}_week_fut'])

    # Split dataset on train, validation and test
    Y = data[f'target_{lag}_week_fut']
    X = data.drop(columns = data.columns[data.columns.str.contains('_week_fut')])
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(
        X, Y, test_size = test_size, random_state = random_state)
    X_train, X_val, Y_train, Y_val = sk.model_selection.train_test_split(
        X_train, Y_train, test_size = (1 - test_size) * val_size, random_state = random_state)

    # Fixed hyperparams for the models
    param_lgb = {
        "verbosity": -1,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": random_state,
        # "device": "gpu",
        # "gpu_platform_id": 0,
        # "gpu_device_id": 0
    }

    param_xgb = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "random_state": random_state
        # "device": "gpu"
    }

    param_cat = {
        "verbose": False,
        "loss_function": "RMSE",
        "bootstrap_type": "Bayesian",
        "random_seed": random_state,
        # "task_type": "GPU", 
        # "devices": "0:1"
    }

    train_param_lxgb = {
        "num_boost_round": 500,
        "early_stopping_rounds": 100,
        "verbose_eval": 100
    }

    train_param_cat = {
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    dirs = {
        'lgb': directory + f'Models/{lag}/lgb.txt',
        'xgb': directory + f'Models/{lag}/xgb.json',
        'cat': directory + f'Models/{lag}/cat'
    }

    # Find optimal hyperparams for LightGBM with Optuna
    print(f'\n LightGBM, {lag} lag:')
    trial_lgb = optuna_study('lgb')
    trial_lgb.params.update(param_lgb)

    # Train LightGBM model
    train_pred_lgb, val_pred_lgb, test_pred_lgb, score_lgb, shap_lgb = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "lgb", trial_lgb.params, train_param_lxgb, 
        dirs['lgb'], log = log, shaps = shaps)
    
    # Calculate SHAP values for LightGBM model
    if shaps == True:
        shap.plots.beeswarm(shap_lgb, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/lgb.png', bbox_inches = 'tight', dpi = 750)
    
    # Find optimal hyperparams for XGBoost with Optuna
    print(f'\n XGBoost, {lag} lag:')
    trial_xgb = optuna_study('xgb')
    trial_xgb.params.update(param_xgb)

    # Train XGBoost model
    train_pred_xgb, val_pred_xgb, test_pred_xgb, score_xgb, shap_xgb = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "xgb", trial_xgb.params, train_param_lxgb, 
        dirs['xgb'], log = log, shaps = shaps)
    
    # Calculate SHAP values for XGBoost model
    if shaps == True:
        shap.plots.beeswarm(shap_xgb, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/xgb.png', bbox_inches = 'tight', dpi = 750)
    
    # Find optimal hyperparams for CatBoost with Optuna
    print(f'\n CatBoost, {lag} lag:')
    trial_cat = optuna_study('cat')
    trial_cat.params.update(param_cat)

    # Train CatBoost model
    train_pred_cat, val_pred_cat, test_pred_cat, score_cat, shap_cat = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "cat", trial_cat.params, train_param_cat, 
        dirs['cat'], log = log, shaps = shaps)
    
    # Calculate SHAP values for CatBoost model
    if shaps == True:
        shap.plots.beeswarm(shap_cat, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/cat.png', bbox_inches = 'tight', dpi = 750)

    # Create comparison table
    stats = pd.DataFrame()
    stats['models'] = ['lgb', 'xgb', 'cat']
    stats['valid'] = [score_lgb["val"], score_xgb["val"], score_cat["val"]]
    stats['test'] = [score_lgb["test"], score_xgb["test"], score_cat["test"]]

    # Create table with target and predictions for validation
    stack_val = pd.DataFrame()
    stack_val['orig'] = Y_val
    stack_val['lgb'] = val_pred_lgb
    stack_val['xgb'] = val_pred_xgb
    stack_val['cat'] = val_pred_cat

    # Create table with target and predictions for test
    stack_test = pd.DataFrame()
    stack_test['orig'] = Y_test
    stack_test['lgb'] = test_pred_lgb
    stack_test['xgb'] = test_pred_xgb
    stack_test['cat'] = test_pred_cat

    # Train stacking model on the validation data
    print(f'\n Stacking, {lag} lag:')
    results, val_rmse, test_rmse, Y_val_pred, Y_test_pred = OLS_optimization(
        stack_val.orig, stack_val.drop('orig', axis = 1).copy(), 
        stack_test.orig, stack_test.drop('orig', axis = 1).copy(), log = log)
    results.save(directory + f'Models/{lag}/stacking.pickle')

    # Update comparison table with stacked results and print it
    stats.loc[len(stats)] = ['stacked', val_rmse, test_rmse]
    print(f'\n Comparison, {lag} lag:')
    print(stats)

    # Save target and predictions for validation and tests
    stack_val['stack'] = Y_val_pred
    stack_test['stack'] = Y_test_pred
    stack_val.to_parquet(directory + f'Predictions/{lag}/gb_val.parquet')
    stack_test.to_parquet(directory + f'Predictions/{lag}/gb_test.parquet')

#---------------------------------------------------------------------------------------------------------------------------------------

def target_pred_dist(lag:int,
                     directory:str = ''):

    """
    Function for the plotting of the distributions of target and predictions

    Inputs:
    ----------
    lag : int
        Distance of prediction in weeks
    directory : str = ''
        Directory where data is stored if it isn't CWD

    Files:
    ----------
    Plot with comparison of validation target and predictions
    Plot with comparison of test target and predictions
    """

    # Import log flag from config
    config.read(directory + 'config.cfg')
    log = bool(config.get('params', 'log'))

    samples = {'val': 'Validation', 'test': 'Test'}
    for key in samples.keys():
        # Import validation and test target and predictions
        data = pd.read_parquet(directory + f'Predictions/{lag}/gb_{key}.parquet')[['stack', 'orig']]
        if log == True:
            data = np.exp(data)

        # Print bucket analysis
        res_buckets = pd.DataFrame(columns = ['Lower', 'Upper', 'Number', 'RMSE', 'MAE'])
        buckets = np.linspace(data['orig'].min(), data['orig'].max(), 11)
        for i, bucket in enumerate(buckets[:-1]):
            stack_bucket = data[data['orig'].between(bucket, buckets[i + 1])]
            res_buckets.loc[len(res_buckets)] = [bucket, buckets[i + 1], len(stack_bucket), 
                                                 mse(stack_bucket['orig'], stack_bucket['stack'], squared = False),
                                                 mae(stack_bucket['orig'], stack_bucket['stack'])]
        print(f'\n {samples[key]} buckets, {lag} lag:')
        print(res_buckets)

        # Plot data distributions regarding if it was logged before
        fig = go.Figure()
        fig.add_trace(go.Histogram(x = data['orig'], name = f'{samples[key]} target'))
        fig.add_trace(go.Histogram(x = data['stack'], name = f'{samples[key]} stacked prediction'))
        fig.update_layout(barmode = 'overlay',
                          showlegend = True,
                          font = dict(size = 30),
                          title = f'{samples[key]} Predictions vs Target for {lag} weeks',
                          title_x = 0.5,
                          xaxis_title = 'Home price, $',
                          yaxis_title = 'Count',
                          legend = dict(x = 0.8, y = 1, traceorder = 'normal'))
        fig.update_traces(opacity = 0.75)
        fig.update_layout()

        # Save plots
        pio.write_image(fig, directory + f"Predictions/{lag}/{key}_dist.png", scale = 6, width = 3000, height = 1500)
        pio.write_image(fig, directory + f"Predictions/{lag}/{key}_dist.svg", scale = 6, width = 3000, height = 1500)

        # Plot distributions of errors
        fig = go.Figure()
        fig.add_trace(go.Histogram(x = data['orig'] - data['stack'], name = f'{samples[key]} errors'))
        fig.update_layout(showlegend = True,
                          font = dict(size = 30),
                          title = f'{samples[key]} Errors for {lag} weeks',
                          title_x = 0.5,
                          xaxis_title = 'Error, $',
                          yaxis_title = 'Count',
                          legend = dict(x = 0.8, y = 1, traceorder = 'normal'))
        fig.update_layout()

        # Save plots
        pio.write_image(fig, directory + f"Predictions/{lag}/{key}_error_dist.png", scale = 6, width = 3000, height = 1500)
        pio.write_image(fig, directory + f"Predictions/{lag}/{key}_error_dist.svg", scale = 6, width = 3000, height = 1500)

#---------------------------------------------------------------------------------------------------------------------------------------

def emd(signal, 
        t, 
        plot:bool = False):

    """
    Function for the decomposition of time series to the several components until the last one is monotonous
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    plot : bool = False
        Flag whether to plot of decomposed time series is needed

    Plots:
    ----------
    Plots of original time series and its decomposed parts if plot == True

    Returns:
    ----------
    imfs : array
        Decomposed time series
    """

    # Separate time series into components
    emd = EMD(DTYPE = np.float16, spline_kind = 'akima')
    imfs = emd(signal.values)
    N = imfs.shape[0]
    
    if plot:
        # Creating grid of subplots
        fig = make_subplots(rows = N + 1, cols = 1, subplot_titles = ['Original Signal'] + [f'IMF {i}' for i in range(N)])

        # Scattering signal and IMFs
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        for i, imf in enumerate(imfs):
            fig.add_trace(go.Scatter(x = t, y = imf, mode = 'lines', name = f'IMF {i}'), row = i + 2, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 400 * (N + 1),
            width = 2000
        )
        fig.show()

    return imfs

#---------------------------------------------------------------------------------------------------------------------------------------

def phase_spectrum(imfs):

    """
    Function for the calculation of the time series' phase spectrum
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    imfs : array
        Decomposed time series

    Returns:
    ----------
    imfs_p : array
        Phase spectrum of decomposed time series
    """

    # Iterate over decomposed timer series to calculate each ones phase spectrum
    imfs_p = []
    for imf in imfs:
        trans = fft(imf)
        imf_p = np.arctan(trans.imag / trans.real)
        imfs_p.append(imf_p)

    return imfs_p

#---------------------------------------------------------------------------------------------------------------------------------------

def phase_mi(phases):

    """
    Function for the calculation of mutual information in the phases
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    phases : array
        Phase spectrum of decomposed time series
    
    Returns:
    ----------
    mis : array
        Mutual information of phase spectrums of decomposed time series
    """

    # Iterate over phases to calculate mutual info
    mis = []
    for i in range(len(phases) - 1):
        mis.append(mutual_info_regression(phases[i].reshape(-1, 1), phases[i + 1])[0])
        
    return np.array(mis)

#---------------------------------------------------------------------------------------------------------------------------------------

def divide_signal(signal, 
                  t, 
                  imfs, 
                  mis, 
                  cutoff:float = 0.05, 
                  plot = False):

    """
    Function for the final separation to the stohastic and determenistic components
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    imfs : array
        Decomposed time series
    mis : array
        Mutual information of phase spectrums of decomposed time series
    cutoff : float = 0.05
        Border of separation between stohastic and determenistic components
    plot : bool = False
        Flag whether to plot original time series, stohastic and determenistic components

    Plots:
    ----------
    Plots of original time series, stohastic and determenistic components if plot == True

    Returns:
    ----------
    stochastic_component : array
        Sum of time series components that are considered stohastic
    deterministic_component : array
        Sum of time series components that are considered deterministic
    """

    # Separate time series to stohastic and deterministic components 
    cut_point = np.where(mis > cutoff)[0][0]    
    stochastic_component = np.sum(imfs[:cut_point], axis=0)
    deterministic_component = np.sum(imfs[cut_point:], axis=0)

    if plot:
        # Creating grid of subplots
        fig = make_subplots(rows = 3, cols = 1, subplot_titles = ['Original Signal', 'Stochastic Component', 'Deterministic Component'])
        
        # Scattering signal and components
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        fig.add_trace(go.Scatter(x = t, y = stochastic_component, mode = 'lines', name = 'Stochastic Component'), row = 2, col = 1)
        fig.add_trace(go.Scatter(x = t, y = deterministic_component, mode = 'lines', name = 'Deterministic Component'), row = 3, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 1200,
            width = 2000
        )
        fig.show()
    
    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def check_2008(lag:int,
               periods:list = None,
               directory:str = '', 
               smooth:bool = True):

    """
    Function for the additional testing with Case-Shiller index

    Inputs:
    ----------
    lag : int
        Distance of prediction in weeks
    periods : list = None
        Periods to analyze separately. Should look like [[start_1, finish_1], [start_2, finish_2]]. Borders are included
    directory : str = ''
        Directory where data is stored if it isn't CWD
    smooth : bool = True
        Flag whether to smooth stacked prediction with FFT
    
    Files:
    ----------
    Plot of comparison between target and (smoothed) prediction
    """

    # Dict with directories where models are stored
    dirs = {
        'lgb': directory + f'Models/{lag}/lgb.txt',
        'xgb': directory + f'Models/{lag}/xgb.json',
        'cat': directory + f'Models/{lag}/cat'
    }

    # Import log flag from config
    config.read(directory + 'config.cfg')
    log = bool(config.get('params', 'log'))

    # Import Case-Shiller data
    data = pd.read_parquet(directory + 'Data_for_models/final_CS.parquet').dropna(subset = [f'target_{lag}_week_fut'])

    # Import boosting and stacking models
    lgb_model = lgb.Booster(model_file = dirs['lgb'])
    xgb_model = xgb.Booster()
    xgb_model.load_model(dirs['xgb'])
    cat_model = cat.CatBoost().load_model(dirs['cat'])
    stacking_model = sm.load(directory + f'Models/{lag}/stacking.pickle')

    # Split dataset on train, validation and test
    train_data = pd.read_parquet(directory + 'Data_for_models/final_full.parquet')
    train_cols = train_data.drop(columns = train_data.columns[train_data.columns.str.contains('_week_fut')])

    # Split data to X and Y
    Y = data[f'target_{lag}_week_fut']
    X = data.drop(columns = data.columns[data.columns.str.contains('_week_fut')])
    X = X[train_cols.columns]
    X_xgb = xgb.DMatrix(X)

    # Predict target with all models
    pred_lgb = lgb_model.predict(X)
    pred_xgb = xgb_model.predict(X_xgb)
    pred_cat = cat_model.predict(X)
    preds = pd.DataFrame({'orig': Y, 'lgb': pred_lgb, 'xgb': pred_xgb, 'cat': pred_cat})
    preds['stack'] = stacking_model.predict(preds[list(stacking_model.params.index)])

    # Smooth stacked prediction with FFT
    if smooth == True:
        imfs = emd(preds['stack'], preds.index)
        imfs_p = phase_spectrum(imfs)
        mis = phase_mi(imfs_p)
        rmse_min = np.inf
        best_cut = 0.5
        for cut in np.linspace(0.25, 3, 12):
            try:
                stochastic_component, deterministic_component = divide_signal(preds['stack'], preds.index, imfs, mis, cutoff = cut)
                if mse(deterministic_component, preds['orig'], squared = False) < rmse_min:
                    best_cut = cut
            except:
                pass
        stochastic_component, deterministic_component = divide_signal(preds['stack'], preds.index, imfs, mis, cutoff = best_cut)
        preds['smoothed'] = deterministic_component

    # Save predictions
    if log == True:
        preds = np.exp(preds)
    preds.to_parquet(directory + f'Predictions/{lag}/2008.parquet')
    
    # Print statistics for each of the periods
    print(f'\n 2008 stacking, {lag} lag:')
    if periods == None:
        periods = [[preds.index.min(), preds.index.max()]]
    for period in periods:
        preds_period = preds.loc[period[0]:period[1]]
        print(f'\n Start date : {str(period[0])[:10]}, end date: {str(period[1])[:10]}')
        print('Final RMSE for Case-Shiller:', round(mse(preds_period['stack'], 
                                                        preds_period['orig'], 
                                                        squared = False), 3))
        if smooth == True:
            print('Final RMSE for Case-Shiller with smoothed predictions is:', round(mse(preds_period['smoothed'], 
                                                                                         preds_period['orig'], 
                                                                                         squared = False), 3))

        # Create a plot of predictions vs target and save it
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['orig'], mode = 'lines', name = 'True values'))
        if smooth == True:
            fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['smoothed'], mode = 'lines', name = 'Smoothed stacked prediction'))
        fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['stack'], mode = 'lines', name = 'Stacked prediction', opacity = 0.4))
        fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['lgb'], mode = 'lines', name = 'LightGBM prediction', opacity = 0.2))
        fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['xgb'], mode = 'lines', name = 'XGBoost prediction', opacity = 0.2))
        fig.add_trace(go.Scatter(x = preds_period.index, y = preds_period['cat'], mode = 'lines', name = 'CatBoost prediction', opacity = 0.2))
        fig.update_layout(showlegend = True,
                        font = dict(size = 30),
                        title = 'Predictions vs Case-Shiller',
                        title_x = 0.5,
                        xaxis_title = 'Date',
                        yaxis_title = 'Home price, $',
                        legend = dict(x = 0, y = 1, traceorder = 'normal'))
        pio.write_image(fig, directory + f"Alternative_test/{lag}/CS_{str(period[0])[:10]}_{str(period[1])[:10]}.png", 
                        scale = 6, width = 3000, height = 1500)
        pio.write_image(fig, directory + f"Alternative_test/{lag}/CS_{str(period[0])[:10]}_{str(period[1])[:10]}.svg", 
                        scale = 6, width = 3000, height = 1500)

#---------------------------------------------------------------------------------------------------------------------------------------

def boosting_hyperparameters(lag:int,
                             directory:str = ''):

    """
    Function for the retrieval of the gradient boosting hyperparameters

    Inputs:
    ----------
    lag : int
        Distance of prediction in weeks
    directory : str = ''
        Directory where data is stored if it isn't CWD

    Prints:
    ----------
    Hyperparameters for LightGBM, XGBoost and Catboost models
    """

    # Dict with directories where models are stored
    dirs = {
        'lgb': directory + f'Models/{lag}/lgb.txt',
        'xgb': directory + f'Models/{lag}/xgb.json',
        'cat': directory + f'Models/{lag}/cat'
    }

    # Get hyperparameters for LightGBM
    lgb_model = lgb.Booster(model_file = dirs['lgb'])
    print(f'\n LightGBM, {lag} lag:')
    print(lgb_model.params)

    # Get hyperparameters for XGBoost
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(dirs['xgb'])
    print(f'\n XGBoost, {lag} lag:')
    print(xgb_model.get_xgb_params())

    # Get hyperparameters for CatBoost
    cat_model = cat.CatBoost().load_model(dirs['cat'])
    print(f'\n CatBoost, {lag} lag:')
    print(cat_model.get_params())