"""
2-parameter model for fitting demand-backup relationship
"""
import numpy as np

def model_2params(x, a, b, N):
    """
    Rational function model with 2 parameters
    
    Parameters:
    -----------
    x : float or array
        Demand share values
    a, b : float
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
        
        # Safely calculate the term
        term = np.power(diff, b)
        
        # Compute the result only for valid inputs
        result[valid_mask] = a * term / (1 + term) + x0
    
    # Handle x < x0 case (set to x0 as a floor)
    result[x <= x0] = x0
    
    # Return the same format as input
    if scalar_input:
        return result[0]
    else:
        return result