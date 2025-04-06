"""
2-parameter model for fitting demand-backup relationship
"""

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
    term = (x - x0)**b  # Calculate (x-1/N)^b once to avoid repetition
    return a * term/(1 + term) + x0