import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
    df_prepared = df.copy()
    
    if 'Demand share' not in df_prepared.columns:
        if 'Demand' in df_prepared.columns:
            max_demand = df_prepared['Demand'].max()
            df_prepared['Demand share'] = df_prepared['Demand'] / max_demand
    
    if 'Back-up share' not in df_prepared.columns:
        if 'Back-up' in df_prepared.columns:
            max_backup = df_prepared['Back-up'].max()
            df_prepared['Back-up share'] = df_prepared['Back-up'] / max_backup
    
    df_prepared = df_prepared.replace([float('inf'), -float('inf')], float('nan'))
    df_prepared = df_prepared.dropna(subset=['Demand', 'Demand share', 'Back-up share'])
    
    df_prepared = df_prepared[df_prepared['Demand'] > 0]
    df_prepared = df_prepared[df_prepared['Demand share'] > 0]
    df_prepared = df_prepared[df_prepared['Back-up share'] > 0]
    
    x0 = 1/N
    df_prepared = df_prepared[df_prepared['Demand share'] >= x0]
    
    return df_prepared

def run_analysis(data_path, output_dir, min_bin_range=10, r2_threshold=0.75, initial_bins=200):

    os.makedirs(output_dir, exist_ok=True)
    
    param_dfs = {}
    
    for N in range(2, 6):
        print(f"\n{'='*50}")
        print(f"Processing {N} loads configuration")
        print(f"{'='*50}")
        
        data_file = os.path.join(data_path, f"data_{N}loads.csv")
        df_data = pd.read_csv(data_file)
        
        df_data = prepare_data(df_data, N)
        
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
        
        param_df = plot_parameters_variation_3params(results)
        
        n_points = len(df_data)
        n_bins = len(results)
        
        if n_bins > 0:
            mean_r2 = param_df['R2'].mean()
            a_mean = param_df['a'].mean()
            a_std = param_df['a'].std()
            b_mean = param_df['b'].mean()
            b_std = param_df['b'].std()
            c_mean = param_df['c'].mean()
            c_std = param_df['c'].std()
            
            print(f"Data points analyzed: {n_points}")
            print(f"Fitted models: {n_bins}")
            print(f"Mean R² score: {mean_r2:.4f}")
            print(f"Parameter a: {a_mean:.4f} ± {a_std:.4f}")
            print(f"Parameter b: {b_mean:.4f} ± {b_std:.4f}")
            print(f"Parameter c: {c_mean:.4f} ± {c_std:.4f}")
        else:
            print(f"Data points analyzed: {n_points}")
            print("No models met the quality threshold")
        n
        param_df['loads'] = N
        param_dfs[N] = param_df
    
    combined_df = pd.concat(list(param_dfs.values()))
    
    print("\nAnalyzing parameter relationships...")
    reliable_params = identify_reliable_parameters(combined_df, param_count=3, threshold_std=2.0)
    
    reliable_output = os.path.join(output_dir, "reliable_params_3p.csv")
    reliable_params.to_csv(reliable_output, index=False)
    print(f"Parameter data saved to {reliable_output}")
    
    print("\nGenerating visualization...")
    plot_parameter_relationships(reliable_params, param_count=3)
    plt.savefig(os.path.join(output_dir, "parameter_relationships_3p.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

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
    
    stats_output = os.path.join(output_dir, "param_stats_3p.csv")
    stats_df.to_csv(stats_output, index=False)
    
    print("\nAnalysis Summary:")
    print(f"Total parameter sets analyzed: {len(combined_df)}")
    print(f"Reliable parameter sets identified: {len(reliable_params)}")
    print(f"Outputs saved to: {output_dir}")
    
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