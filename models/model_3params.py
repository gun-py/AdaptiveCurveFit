"""
3-parameter model for fitting demand-backup relationship
"""
import numpy as np

def model_3params(x, a, b, c, N):
    x0 = 1/N  # Minimum share
    
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        scalar_input = True
    else:
        scalar_input = False
    
    # same as 2 param model script
    result = np.full_like(x, x0, dtype=float)
    
    valid_mask = x > x0
    
    if np.any(valid_mask):
        x_valid = x[valid_mask]
        diff = x_valid - x0
        
        denominator = a * c * diff + b
        # limit of denominator, not too close to zero to avoid zero div
        safe_mask = np.abs(denominator) > 1e-10
        
        if np.any(safe_mask):
            numerator = a * b * c * diff
            result[valid_mask] = np.where(
                safe_mask,
                numerator[safe_mask] / denominator[safe_mask] + x0,
                x0 
            )
    
    # flooring
    result[x <= x0] = x0
    
    result = np.maximum(result, x0)
    
    if scalar_input:
        return result[0]
    else:
        return result