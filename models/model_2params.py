"""
2-parameter model for fitting demand-backup relationship
"""
import numpy as np

def model_2params(x, a, b, N):
    x0 = 1/N 
    
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        scalar_input = True
    else:
        scalar_input = False
    
    # (safe default)
    result = np.full_like(x, x0, dtype=float)
    
    # edge cases for NaN
    valid_mask = x > x0
    
    if np.any(valid_mask):
        x_valid = x[valid_mask]
        diff = x_valid - x0
        
        term = np.power(diff, b)
        
        result[valid_mask] = a * term / (1 + term) + x0
    
    result[x <= x0] = x0
    
    if scalar_input:
        return result[0]
    else:
        return result