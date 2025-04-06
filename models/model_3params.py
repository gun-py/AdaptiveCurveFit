"""
3-parameter model for fitting demand-backup relationship
"""
import numpy as np

def model_3params(x, a, b, c, N):
    """
    Modified saturation model with fraction form (3 parameters)
    
    Parameters:
    -----------
    x : float or array
        Demand share values
    a, b, c : float
        Model parameters
    N : int
        Number of systems
        
    Returns:
    --------
    float or array
        Back-up share values
    """
    x0 = 1/N  # Minimum share
    
    # Convert to numpy array if it's not already
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        scalar_input = True
    else:
        scalar_input = False
    
    # Initialize result array with x0 (safe default)
    result = np.full_like(x, x0, dtype=float)
    
    # Create a mask for values where x > x0 (safe for calculation)
    valid_mask = x > x0
    
    if np.any(valid_mask):
        # Only calculate for valid values
        x_valid = x[valid_mask]
        diff = x_valid - x0
        
        # Handle potential division by zero or overflow
        denominator = a * c * diff + b
        # Ensure denominator is not too close to zero
        safe_mask = np.abs(denominator) > 1e-10
        
        if np.any(safe_mask):
            # Safely compute only where denominator is not too small
            numerator = a * b * c * diff
            result[valid_mask] = np.where(
                safe_mask,
                numerator[safe_mask] / denominator[safe_mask] + x0,
                x0  # Default to x0 where calculation would be unstable
            )
    
    # Handle edge cases (set to x0 as a floor)
    result[x <= x0] = x0
    
    # Ensure all values are at least x0
    result = np.maximum(result, x0)
    
    # Return the same format as input
    if scalar_input:
        return result[0]
    else:
        return result