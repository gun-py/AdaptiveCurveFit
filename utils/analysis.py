"""
Analysis utilities for demand-backup models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

def analyze_demand_bins(df_data, N, min_bin_range, r2_threshold, initial_bins, 
                        fit_func, plot_func, model_func, param_count=3):
    """
    Analyze demand data with dynamic binning
    
    Parameters:
    -----------
    df_data : pandas.DataFrame
        DataFrame with 'Demand' and 'Back-up share' columns
    N : int
        Number of systems
    min_bin_range : float
        Minimum range for a bin
    r2_threshold : float
        Threshold for R² score
    initial_bins : int
        Initial number of bins
    fit_func : function
        Function for fitting
    plot_func : function
        Function for plotting
    model_func : function
        Model function
    param_count : int
        Number of parameters in the model (2 or 3)
        
    Returns:
    --------
    list
        List of dictionaries with results for each bin
    """
    min_val = df_data['Demand'].min()
    max_val = df_data['Demand'].max()
    range_val = max_val - min_val

    # Initial binning
    class_size = max(range_val / initial_bins, min_bin_range)
    bins = [min_val + i * class_size for i in range(int(range_val/class_size) + 1)]

    results = []

    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        extension = 0
        r2 = 0

        while r2 < r2_threshold and extension < class_size:
            # Extend bin boundaries
            extended_start = max(min_val, bin_start - extension)
            extended_end = min(max_val, bin_end + extension)

            # Get data for current bin
            df_subset = df_data[
                (df_data['Demand'] >= extended_start) &
                (df_data['Demand'] <= extended_end)
            ].copy()

            # Fit model and get R²
            params, fitted_df, r2 = fit_func(df_subset, extended_start, extended_end, N, model_func)

            if r2 >= r2_threshold:
                # Save the fit if it meets the R² threshold
                results.append({
                    'bin_start': extended_start,
                    'bin_end': extended_end,
                    'r2': r2,
                    'params': params,
                    'n_points': len(df_subset)
                })
                break

            extension += class_size * 0.1

    return results

def identify_reliable_parameters(df, param_count=3, threshold_std=2.0):
    """
    Identify reliable parameter estimates by filtering outliers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with parameters
    param_count : int
        Number of parameters (2 or 3)
    threshold_std : float
        Threshold for standardized residuals
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with reliable parameter estimates
    """
    total_points = len(df)

    # Calculate adaptive parameter ranges based on data statistics
    def calculate_param_bounds(param_series):
        """Calculate lower and upper bounds for parameter using IQR method"""
        q1 = param_series.quantile(0.25)
        q3 = param_series.quantile(0.75)
        iqr = q3 - q1
        
        # Use 2.0 as IQR multiplier for a moderately conservative approach
        lower_bound = max(0, q1 - 2.0 * iqr)  # Parameters are generally non-negative
        upper_bound = q3 + 2.0 * iqr
        
        return lower_bound, upper_bound
    
    # Calculate bounds for each parameter
    a_bounds = calculate_param_bounds(df['a'])
    b_bounds = calculate_param_bounds(df['b'])
    
    # Identify parameters within expected range
    if param_count == 3:
        c_bounds = calculate_param_bounds(df['c'])
        
        df_valid = df[
            (df['a'].between(*a_bounds)) &
            (df['b'].between(*b_bounds)) &
            (df['c'].between(*c_bounds))
        ].copy()
    else:  # param_count == 2
        df_valid = df[
            (df['a'].between(*a_bounds)) &
            (df['b'].between(*b_bounds))
        ].copy()

    valid_param_count = len(df_valid)

    # Identify consistent parameter-demand relationships
    def identify_relationship_outliers(x, y, threshold):
        """Identify points with consistent parameter-demand relationships"""
        X = x.values.reshape(-1, 1)
        Y = y.values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, Y)

        y_pred = model.predict(X)
        residuals = Y - y_pred

        std_residuals = np.abs(stats.zscore(residuals))
        return std_residuals.reshape(-1) < threshold

    # Find consistent parameter patterns
    mask_a = identify_relationship_outliers(df_valid['Demand'], df_valid['a'], threshold_std)
    mask_b = identify_relationship_outliers(df_valid['Demand'], df_valid['b'], threshold_std)
    
    if param_count == 3:
        mask_c = identify_relationship_outliers(df_valid['Demand'], df_valid['c'], threshold_std)
        final_mask = mask_a & mask_b & mask_c
    else:
        final_mask = mask_a & mask_b
        
    df_reliable = df_valid[final_mask].copy()
    
    # Print progress information
    print(f"\nIdentified {len(df_reliable)} consistent parameter sets")
    print("-" * 60)
    
    # Fit linear models to the reliable parameter sets
    def fit_parameter_model(x, y):
        X = x.values.reshape(-1, 1)
        Y = y.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, Y)
        return model, model.score(X, Y)
    
    # Fit models for each parameter
    model_a, r2_a = fit_parameter_model(df_reliable['Demand'], df_reliable['a'])
    model_b, r2_b = fit_parameter_model(df_reliable['Demand'], df_reliable['b'])
    
    if param_count == 3:
        model_c, r2_c = fit_parameter_model(df_reliable['Demand'], df_reliable['c'])
    
    # Print model coefficients
    print("\nParameter-Demand Relationships:")
    print(f"a = {model_a.coef_[0][0]:.4f} * Demand + {model_a.intercept_[0]:.4f} (R² = {r2_a:.4f})")
    print(f"b = {model_b.coef_[0][0]:.4f} * Demand + {model_b.intercept_[0]:.4f} (R² = {r2_b:.4f})")
    if param_count == 3:
        print(f"c = {model_c.coef_[0][0]:.4f} * Demand + {model_c.intercept_[0]:.4f} (R² = {r2_c:.4f})")
    
    # Create final output DataFrame with the reliable parameter sets
    output_df = df_reliable[['Demand', 'a', 'b', 'loads']].copy()
    if param_count == 3:
        output_df['c'] = df_reliable['c']
    
    return output_df