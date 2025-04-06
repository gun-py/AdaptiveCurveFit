#!/usr/bin/env python
"""
Script to run the 3-parameter model analysis
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import local packages
from models import model_3params
from utils import (
    fit_and_evaluate_bin_3params,
    plot_fit_3params,
    plot_parameters_variation_3params,
    analyze_demand_bins,
    identify_reliable_parameters,
    plot_parameter_relationships
)

def prepare_data(df, N):
    """
    Prepare data for analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe
    N : int
        Number of systems
        
    Returns:
    --------
    pandas.DataFrame
        Prepared dataframe
    """
    # Make a copy to avoid modifying the original
    df_prepared = df.copy()
    
    # Ensure we have the necessary columns
    if 'Demand share' not in df_prepared.columns:
        # Calculate demand share if not already in data
        if 'Demand' in df_prepared.columns:
            max_demand = df_prepared['Demand'].max()
            df_prepared['Demand share'] = df_prepared['Demand'] / max_demand
    
    if 'Back-up share' not in df_prepared.columns:
        # Calculate backup share if not already in data
        if 'Back-up' in df_prepared.columns:
            max_backup = df_prepared['Back-up'].max()
            df_prepared['Back-up share'] = df_prepared['Back-up'] / max_backup
    
    return df_prepared

def run_analysis(data_path, output_dir, min_bin_range=10, r2_threshold=0.75, initial_bins=200):
    """
    Run the 3-parameter model analysis
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the data directory
    output_dir : str or Path
        Path to save outputs
    min_bin_range : float
        Minimum range for bins
    r2_threshold : float
        R² threshold for acceptable fits
    initial_bins : int
        Number of initial bins
        
    Returns:
    --------
    dict
        Dictionary with parameter DataFrames for each load configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store parameter DataFrames
    param_dfs = {}
    
    # Process each load configuration (2, 3, 4, 5 loads)
    for N in range(2, 6):
        print(f"\n{'='*50}")
        print(f"Processing {N} loads configuration")
        print(f"{'='*50}")
        
        # Load data
        data_file = os.path.join(data_path, f"data_{N}loads.csv")
        df_data = pd.read_csv(data_file)
        
        # Prepare data
        df_data = prepare_data(df_data, N)
        
        # Analyze demand bins
        results = analyze_demand_bins(
            df_data,
            N=N,
            min_bin_range=min_bin_range,
            r2_threshold=r2_threshold,
            initial_bins=initial_bins,
            fit_func=fit_and_evaluate_bin_3params,
            plot_func=plot_fit_3params,
            model_func=model_3params,
            param_count=3
        )
        
        # Get parameter DataFrame
        param_df = plot_parameters_variation_3params(results)
        
        # Save parameter DataFrame
        param_df['loads'] = N
        param_dfs[N] = param_df
        
        # Save parameter DataFrame to CSV
        output_file = os.path.join(output_dir, f"params_3p_{N}loads.csv")
        param_df.to_csv(output_file, index=False)
        print(f"Parameters saved to {output_file}")
    
    # Combine all parameter DataFrames
    combined_df = pd.concat(list(param_dfs.values()))
    combined_output = os.path.join(output_dir, "combined_params_3p.csv")
    combined_df.to_csv(combined_output, index=False)
    print(f"\nCombined parameters saved to {combined_output}")
    
    # Identify reliable parameter estimates
    print("\nIdentifying reliable parameter estimates...")
    reliable_params = identify_reliable_parameters(combined_df, param_count=3, threshold_std=2.0)
    
    # Save reliable parameters
    reliable_output = os.path.join(output_dir, "reliable_params_3p.csv")
    reliable_params.to_csv(reliable_output, index=False)
    print(f"Reliable parameter estimates saved to {reliable_output}")
    
    # Plot final parameter relationships
    print("\nPlotting parameter relationships...")
    plot_parameter_relationships(reliable_params, param_count=3)
    plt.savefig(os.path.join(output_dir, "parameter_relationships_3p.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate parameter statistics
    print("\nParameter Statistics:")
    stats_df = pd.DataFrame({
        'Parameter': ['a', 'b', 'c'],
        'Mean': [
            reliable_params['a'].mean(),
            reliable_params['b'].mean(),
            reliable_params['c'].mean()
        ],
        'Std': [
            reliable_params['a'].std(),
            reliable_params['b'].std(),
            reliable_params['c'].std()
        ],
        'Min': [
            reliable_params['a'].min(),
            reliable_params['b'].min(),
            reliable_params['c'].min()
        ],
        'Max': [
            reliable_params['a'].max(),
            reliable_params['b'].max(),
            reliable_params['c'].max()
        ]
    })
    print(stats_df.round(4).to_string(index=False))
    
    # Save statistics
    stats_output = os.path.join(output_dir, "param_stats_3p.csv")
    stats_df.to_csv(stats_output, index=False)
    print(f"Parameter statistics saved to {stats_output}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total bins analyzed: {sum(len(results) for N, results in param_dfs.items())}")
    print(f"Total parameter sets: {len(combined_df)}")
    print(f"Reliable parameter sets: {len(reliable_params)}")
    print(f"Final outputs saved to: {output_dir}")
    
    return reliable_params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 3-parameter model analysis")
    parser.add_argument("--data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="results_3param", help="Path to output directory")
    parser.add_argument("--min-bin-range", type=float, default=10, help="Minimum bin range")
    parser.add_argument("--r2-threshold", type=float, default=0.75, help="R² threshold")
    parser.add_argument("--initial-bins", type=int, default=200, help="Number of initial bins")
    
    args = parser.parse_args()
    
    run_analysis(
        args.data,
        args.output,
        min_bin_range=args.min_bin_range,
        r2_threshold=args.r2_threshold,
        initial_bins=args.initial_bins
    )