"""
3-parameter model for fitting demand-backup relationship
"""

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
    return (a*b*c*(x - x0))/(a*c*(x - x0) + b) + x0