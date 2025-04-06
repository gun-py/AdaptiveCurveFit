import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_fit_3params(df_plot, params, min_demand, max_demand, r2, N, model_func):
    if df_plot is None or params is None:
        return

    x0 = 1/N

    # smooth curve
    x_smooth = np.linspace(df_plot['Demand share'].min(), df_plot['Demand share'].max(), 100)
    
    # Wrapper model function
    def model_wrapper(x):
        return model_func(x, params[0], params[1], params[2], N)
    
    y_smooth = model_wrapper(x_smooth)

    plt.figure(figsize=(10, 5))
    plt.scatter(df_plot['Demand share'], df_plot['Back-up share'], label="Data points")
    plt.plot(
        x_smooth,
        y_smooth,
        '--',
        label=f"y = (a·b·c(x-x₀))/(a·c(x-x₀)+b) + x₀\na={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}\nR²: {r2:.4f}" # This is dummy
    )
    plt.scatter([x0], [x0], color='black')
    plt.title(f'{min_demand:.1f} W < Demand < {max_demand:.1f} W')
    plt.xlim([0, 1]) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    plt.grid()
    plt.xlabel('Demand share')
    plt.ylabel('Back-up share')
    plt.legend()
    plt.tight_layout()

def plot_fit_2params(df_plot, params, min_demand, max_demand, r2, N, model_func):
    if df_plot is None or params is None:
        return

    x0 = 1/N

    x_smooth = np.linspace(df_plot['Demand share'].min(), df_plot['Demand share'].max(), 100)
    
    def model_wrapper(x):
        return model_func(x, params[0], params[1], N)
    
    y_smooth = model_wrapper(x_smooth)

    plt.figure(figsize=(10, 5))
    plt.scatter(df_plot['Demand share'], df_plot['Back-up share'], label="Data points")
    plt.plot(
        x_smooth,
        y_smooth,
        '--',
        label=f"y = a(x-x₀)^b/(1+(x-x₀)^b) + x₀\na={params[0]:.4f}, b={params[1]:.4f}\nR²: {r2:.4f}"
    )
    plt.scatter([x0], [x0], color='black')
    plt.title(f'{min_demand:.1f} W < Demand < {max_demand:.1f} W')
    plt.xlim([0, 1]) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    plt.grid()
    plt.xlabel('Demand share')
    plt.ylabel('Back-up share')
    plt.legend()
    plt.tight_layout()

def plot_parameters_variation_3params(results):
    # mean demand values and parameters
    demands = [(res['bin_start'] + res['bin_end'])/2 for res in results if res['params'] is not None]
    a_values = [res['params'][0] for res in results if res['params'] is not None]
    b_values = [res['params'][1] for res in results if res['params'] is not None]
    c_values = [res['params'][2] for res in results if res['params'] is not None]
    r2_values = [res['r2'] for res in results if res['params'] is not None]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

    ax1.scatter(demands, a_values)
    ax1.set_xlabel('Demand (W)')
    ax1.set_ylabel('Parameter a')
    ax1.set_ylim(0, 20) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    ax1.grid(True)
    ax1.set_title('Variation of Parameter a with Demand')

    ax2.scatter(demands, b_values)
    ax2.set_xlabel('Demand (W)')
    ax2.set_ylabel('Parameter b')
    ax2.set_ylim(0, 10) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    ax2.grid(True)
    ax2.set_title('Variation of Parameter b with Demand')

    ax3.scatter(demands, c_values)
    ax3.set_xlabel('Demand (W)')
    ax3.set_ylabel('Parameter c')
    ax3.set_ylim(0, 1) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    ax3.grid(True)
    ax3.set_title('Variation of Parameter c with Demand')

    ax4.scatter(demands, r2_values)
    ax4.set_xlabel('Demand (W)')
    ax4.set_ylabel('R² Score')
    ax4.grid(True)
    ax4.set_title('R² Score Variation with Demand')

    plt.tight_layout()

    return pd.DataFrame({
        'Demand': demands,
        'a': a_values,
        'b': b_values,
        'c': c_values,
        'R2': r2_values
    })

def plot_parameters_variation_2params(results):
    demands = [(res['bin_start'] + res['bin_end'])/2 for res in results if res['params'] is not None]
    a_values = [res['params'][0] for res in results if res['params'] is not None]
    b_values = [res['params'][1] for res in results if res['params'] is not None]
    r2_values = [res['r2'] for res in results if res['params'] is not None]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

    ax1.scatter(demands, a_values)
    ax1.set_xlabel('Demand (W)')
    ax1.set_ylabel('Parameter a')
    ax1.set_ylim(0, 10) # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
    ax1.grid(True)
    ax1.set_title('Variation of Parameter a with Demand')

    ax2.scatter(demands, b_values)
    ax2.set_xlabel('Demand (W)')
    ax2.set_ylabel('Parameter b')
    ax2.grid(True)
    ax2.set_title('Variation of Parameter b with Demand')

    ax3.scatter(demands, r2_values)
    ax3.set_xlabel('Demand (W)')
    ax3.set_ylabel('R² Score')
    ax3.grid(True)
    ax3.set_title('R² Score Variation with Demand')

    plt.tight_layout()

    return pd.DataFrame({
        'Demand': demands,
        'a': a_values,
        'b': b_values,
        'R2': r2_values
    })

def plot_parameter_relationships(df_params, param_count=3):

    if param_count == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # color mapping
        unique_loads = np.unique(df_params['loads'])
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_loads)))
        
        # 
        for ax, param, ylabel, ylim in zip(
            axes, 
            ['a', 'b', 'c'], 
            ['Parameter a', 'Parameter b', 'Parameter c'],
            [(0, 20), (0, 2), (0, 1)] # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
        ):
            for i, load in enumerate(unique_loads):
                mask = df_params['loads'] == load
                ax.scatter(
                    df_params.loc[mask, 'Demand'],
                    df_params.loc[mask, param],
                    color=colors[i],
                    label=f'N={load}',
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=0.5,
                    s=60
                )
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('Demand (W)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(ylim)
            ax.set_title(f'Demand vs {param}', fontsize=14)
            
        axes[0].legend(title='Number of Systems', fontsize=10)
        
    else:  # param_count == 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        unique_loads = np.unique(df_params['loads'])
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_loads)))
        
        for ax, param, ylabel, ylim in zip(
            axes, 
            ['a', 'b'], 
            ['Parameter a', 'Parameter b'],
            [(0, 5), (0, 5)] # CHANGE THIS ACCORDINGLY !!!!!!!!!!!!!!!!!!!
        ):
            for i, load in enumerate(unique_loads):
                mask = df_params['loads'] == load
                ax.scatter(
                    df_params.loc[mask, 'Demand'],
                    df_params.loc[mask, param],
                    color=colors[i],
                    label=f'N={load}',
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=0.5,
                    s=60
                )
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('Demand (W)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(ylim)
            ax.set_title(f'Demand vs {param}', fontsize=14)
            
        axes[0].legend(title='Number of Systems', fontsize=10)
    
    plt.tight_layout()