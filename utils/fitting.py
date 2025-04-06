"""
Fitting utilities for demand-backup models
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fit_and_evaluate_bin_3params(df_subset, min_demand, max_demand, N, model_func):
    """
    Fit 3-parameter model to a bin and return R² score
    
    Parameters:
    -----------
    df_subset : pandas.DataFrame
        DataFrame with 'Demand share' and 'Back-up share' columns
    min_demand : float
        Minimum demand in the bin
    max_demand : float
        Maximum demand in the bin
    N : int
        Number of systems
    model_func : function
        Model function to fit
        
    Returns:
    --------
    tuple
        (parameters, fitted_dataframe, r2_score)
    """
    if len(df_subset) < 5:  # Minimum points needed for a reliable fit
        return None, None, 0

    # Sort data
    df_subset = df_subset.sort_values(by='Demand share').reset_index(drop=True)

    try:
        # Initial parameter guesses and bounds
        initial_guess = [1, 3, 0]
        bounds = ([0, 0, -1], [50, 50, 1])

        # Fit the model
        def model_wrapper(x, a, b, c):
            return model_func(x, a, b, c, N)
        
        params, _ = curve_fit(
            model_wrapper,
            df_subset['Demand share'],
            df_subset['Back-up share'],
            p0=initial_guess,
            bounds=bounds,
            max_nfev=10000
        )

        # Calculate R² score
        y_pred = model_wrapper(df_subset['Demand share'], *params)
        r2 = r2_score(df_subset['Back-up share'], y_pred)

        return params, df_subset, r2
    except Exception as e:
        print(f"Error in fitting: {e}")
        return None, None, 0

def fit_and_evaluate_bin_2params(df_subset, min_demand, max_demand, N, model_func):
    """
    Fit 2-parameter model to a bin and return R² score
    
    Parameters:
    -----------
    df_subset : pandas.DataFrame
        DataFrame with 'Demand share' and 'Back-up share' columns
    min_demand : float
        Minimum demand in the bin
    max_demand : float
        Maximum demand in the bin
    N : int
        Number of systems
    model_func : function
        Model function to fit
        
    Returns:
    --------
    tuple
        (parameters, fitted_dataframe, r2_score)
    """
    if len(df_subset) < 5:  # Minimum points needed for a reliable fit
        return None, None, 0

    # Sort data
    df_subset = df_subset.sort_values(by='Demand share').reset_index(drop=True)

    try:
        # Initial parameter guesses and bounds
        initial_guess = [1, 3]
        bounds = ([0, 0], [50, 50])

        # Fit the model
        def model_wrapper(x, a, b):
            return model_func(x, a, b, N)
        
        params, _ = curve_fit(
            model_wrapper,
            df_subset['Demand share'],
            df_subset['Back-up share'],
            p0=initial_guess,
            bounds=bounds,
            max_nfev=10000
        )

        # Calculate R² score
        y_pred = model_wrapper(df_subset['Demand share'], *params)
        r2 = r2_score(df_subset['Back-up share'], y_pred)

        return params, df_subset, r2
    except Exception as e:
        print(f"Error in fitting: {e}")
        return None, None, 0