import numpy as np
from scipy.linalg import sqrtm

def introduce_new_pts(new_measurements, sensor, step, unknown_number, unknown_particles, parameters):
    """
    Introduces new potential targets (PTs) based on measurements.
    This implements part of the factor graph shown in Fig. 4, specifically handling the 
    introduction of new targets through measurements y^m and their corresponding factors q^m.
    
    Parameters:
    -----------
    new_measurements : numpy.ndarray
        Matrix of shape (2, num_measurements) containing range and bearing measurements
    sensor : int
        Current sensor index
    step : int
        Current time step k
    unknown_number : float
        Expected number of unknown targets
    unknown_particles : numpy.ndarray
        Particle representation of unknown targets
    parameters : dict
        Dictionary containing model parameters
        
    Returns:
    --------
    tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        - new_pts: Particles for new potential targets
        - new_labels: Labels for new targets [step; sensor; measurement]
        - new_existences: Existence probabilities for new targets
        - xi_messages: ξ messages (corresponds to ξ_m in the factor graph)
    """
    num_particles = parameters['num_particles']
    detection_probability = parameters['detection_probability']
    clutter_intensity = parameters['mean_clutter'] * parameters['clutter_distribution']
    sensor_positions = parameters['sensor_positions'][:, sensor-1]  # -1 for 0-based indexing
    num_measurements = new_measurements.shape[1]

    # Compute unknown intensity (corresponds to intensity of undetected targets)
    # This affects the messages passed through factors q^m in the factor graph
    surveillance_area = ((parameters['surveillance_region'][0, 0] - parameters['surveillance_region'][1, 0]) * 
                        (parameters['surveillance_region'][0, 1] - parameters['surveillance_region'][1, 1]))
    unknown_intensity = unknown_number / surveillance_area
    unknown_intensity *= (1 - detection_probability) ** (sensor - 1)

    # Calculate constants if there are measurements
    if num_measurements:
        constants = calculate_constants_uniform(sensor_positions, new_measurements, 
                                             unknown_particles, parameters)

    # Initialize arrays for new PTs and messages
    new_pts = np.zeros((4, num_particles, num_measurements))
    new_labels = np.zeros((3, num_measurements))
    xi_messages = np.zeros(num_measurements)

    # Process each measurement to create new PTs
    # This corresponds to creating new nodes a^m and b^m in the factor graph
    for measurement in range(num_measurements):
        # Sample new particles based on measurement likelihood
        new_pts[:, :, measurement] = sample_from_likelihood(
            new_measurements[:, measurement], sensor, num_particles, parameters)
        
        # Assign labels [step; sensor; measurement]
        new_labels[:, measurement] = [step, sensor, measurement]
        
        # Compute xi messages (ξ_m in the factor graph)
        # These messages flow from measurement factors to existence variables
        xi_messages[measurement] = 1 + (constants[measurement] * unknown_intensity * 
                                      detection_probability) / clutter_intensity

    # Compute existence probabilities for new targets
    new_existences = xi_messages - 1

    return new_pts, new_labels, new_existences, xi_messages

def calculate_constants_uniform(sensor_position, new_measurements, particles, parameters):
    """
    Calculate normalization constants for the measurement likelihood.
    This implements the likelihood calculations related to the ψ^(i,m) factors in the factor graph.
    
    Parameters:
    -----------
    sensor_position : numpy.ndarray
        Position of the current sensor
    new_measurements : numpy.ndarray
        Matrix of measurements
    particles : numpy.ndarray
        Particle representation of states
    parameters : dict
        Model parameters
        
    Returns:
    --------
    numpy.ndarray
        Normalization constants for each measurement
    """
    meas_var_range = parameters['measurement_variance_range']
    meas_var_bearing = parameters['measurement_variance_bearing']
    num_measurements = new_measurements.shape[1]
    num_particles = particles.shape[1]

    # Calculate constant weight based on surveillance region
    constant_weight = 1 / ((parameters['surveillance_region'][0, 0] - parameters['surveillance_region'][1, 0]) *
                          (parameters['surveillance_region'][0, 1] - parameters['surveillance_region'][1, 1]))

    # Calculate predicted measurements for particles
    predicted_range = np.sqrt(np.sum((particles[:2, :] - 
                                    sensor_position.reshape(2, 1))**2, axis=0))
    predicted_bearing = np.degrees(np.arctan2(particles[0, :] - sensor_position[0],
                                            particles[1, :] - sensor_position[1]))
    
    constant_likelihood = 1 / (2 * np.pi * np.sqrt(meas_var_bearing * meas_var_range))
    
    constants = np.zeros(num_measurements)
    for measurement in range(num_measurements):
        # Compute likelihood for each measurement
        range_term = np.exp(-0.5 * (new_measurements[0, measurement] - predicted_range)**2 / 
                          meas_var_range)
        bearing_term = np.exp(-0.5 * (new_measurements[1, measurement] - predicted_bearing)**2 / 
                            meas_var_bearing)
        constants[measurement] = np.sum(constant_likelihood * range_term * bearing_term / num_particles)

    return constants / constant_weight

def sample_from_likelihood(measurement, sensor_index, num_particles, parameters):
    """
    Sample new particles from the measurement likelihood.
    This implements the sampling related to the ψ^(i,m) factors in the factor graph.
    
    Parameters:
    -----------
    measurement : numpy.ndarray
        Single measurement (range and bearing)
    sensor_index : int
        Index of the current sensor
    num_particles : int
        Number of particles to generate
    parameters : dict
        Model parameters
        
    Returns:
    --------
    numpy.ndarray
        Matrix of sampled particles
    """
    sensor_position = parameters['sensor_positions'][:, sensor_index-1]
    meas_var_range = parameters['measurement_variance_range']
    meas_var_bearing = parameters['measurement_variance_bearing']
    prior_vel_covariance = parameters['prior_velocity_covariance']

    samples = np.zeros((4, num_particles))

    # Sample range and bearing with measurement noise
    random_range = measurement[0] + np.sqrt(meas_var_range) * np.random.randn(num_particles)
    random_bearing = measurement[1] + np.sqrt(meas_var_bearing) * np.random.randn(num_particles)
    
    # Convert to Cartesian coordinates
    samples[0] = sensor_position[0] + random_range * np.sin(np.radians(random_bearing))
    samples[1] = sensor_position[1] + random_range * np.cos(np.radians(random_bearing))
    
    # Sample velocities from prior
    velocity_samples = np.random.randn(2, num_particles)
    samples[2:4] = sqrtm(prior_vel_covariance) @ velocity_samples

    return samples

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