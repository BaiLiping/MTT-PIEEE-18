import numpy as np

def perform_data_association_bp(beta_k, xi_k, check_convergence, threshold, num_iterations):
    """
    Implements the scalable Sum-Product Algorithm (SPA) based Data Association (DA) algorithm.
    
    This implementation follows equations (30) and (31) from the paper, which represent a 
    simplified version of the message passing scheme for binary consistency constraints.
    
    Parameters:
    -----------
    beta_k : numpy.ndarray
        Matrix of shape (num_measurements + 1, num_objects) containing β_k^(i)(m) values
        First row contains β_k^(i)(0) values
    xi_k : numpy.ndarray
        Vector of length num_measurements containing ξ_k^(m) values
    check_convergence : int
        Number of iterations after which to check convergence
    threshold : float
        Convergence threshold for message differences
    num_iterations : int
        Maximum number of iterations to perform
    
    Returns:
    --------
    tuple(numpy.ndarray, numpy.ndarray)
        - phi_k: Matrix of φ_k^[l](i→m) messages
        - v_k: Vector of v_k^[l](m→i) messages
    """
    
    # Get dimensions from input
    num_measurements = beta_k.shape[0] - 1  # Subtract 1 for β_k^(i)(0) row
    num_objects = beta_k.shape[1]
    
    # Initialize output matrices with ones
    phi_k = np.ones((num_measurements, num_objects))
    v_k = np.ones(num_measurements)
    
    # Early return if either dimension is 0
    if num_objects == 0 or num_measurements == 0:
        return phi_k, v_k
    
    # Initialize message matrices
    v_k_current = np.ones((num_measurements, num_objects))  # v_k^[l](m→i)
    
    # Main iteration loop for message passing
    for l in range(num_iterations):
        # Store previous messages for convergence check
        v_k_previous = v_k_current.copy()
        
        # Compute φ_k^[l](i→m) following equation (30)
        # Calculate product term in denominator: Π_{m'≠m} v_k^[l](m'→i)
        beta_v_product = v_k_current * beta_k[1:, :]  # β_k^(i)(m) * v_k^[l](m→i)
        
        # Calculate full denominator: β_k^(i)(0) + Σ_{m'≠m} β_k^(i)(m') * v_k^[l](m'→i)
        # axis=0 means you're operating along the first dimension (rows), resulting in one value per column
        beta_sum = beta_k[0, :] + np.sum(beta_v_product, axis=0)
        
        # Compute phi_k_current (φ_k^[l](i→m)) using broadcasting
        phi_k_current = beta_k[1:, :] / (beta_sum[np.newaxis, :] - beta_v_product)
        
        # Compute v_k^[l](m→i) following equation (31)
        # Calculate denominator: ξ_k^(m)(0) + Σ_{i'≠i} ξ_k^(m)(i') * φ_k^[l-1](i'→m)
        # axis=1 means you're operating along the second dimension (columns), resulting in one value per row
        xi_sum = xi_k + np.sum(phi_k_current, axis=1)
        
        # Update v_k_current (v_k^[l](m→i))
        v_k_current = 1.0 / (xi_sum[:, np.newaxis] - phi_k_current)
        
        # Check convergence every check_convergence iterations
        if (l + 1) % check_convergence == 0:
            # Compute maximum absolute difference in log domain
            distance = np.max(np.abs(np.log(v_k_current/v_k_previous)))
            if distance < threshold:
                break
    
    phi_k = v_k_current
    return phi_k, v_k