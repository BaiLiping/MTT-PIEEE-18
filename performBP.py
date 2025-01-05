import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from copy import deepcopy

@dataclass
class TrackingParams:
    """Parameters for multi-target tracking"""
    num_particles: int = 1000
    detection_probability: float = 0.9
    clutter_density: float = 1e-6  # Expected number of clutter points per unit volume
    mean_clutter: float = 10.0     # Expected number of clutter measurements
    clutter_distribution: float = 1.0  # Uniform clutter distribution
    # Measurement noise parameters
    measurement_variance_range: float = 100.0    # meters^2
    measurement_variance_bearing: float = 0.01   # radians^2
    # Process noise parameters
    process_noise_std: float = 1.0
    # Detection and track management
    existence_threshold: float = 0.5
    # BP parameters
    max_iterations: int = 20
    convergence_threshold: float = 1e-5
    check_convergence: int = 5



def compute_measurement_factors(measurements: np.ndarray,
                              track_states: Dict[int, np.ndarray],
                              sensor_position: np.ndarray,
                              params: TrackingParams) -> np.ndarray:
    """
    Compute measurement factors (v-factors) for belief propagation data association.
    
    The v-factors represent:
    v(z,x) = p(z|x) * p_d / (lambda * c(z))
    where:
    - p(z|x) is measurement likelihood
    - p_d is detection probability
    - lambda is mean number of clutter points
    - c(z) is clutter distribution
    Likelihood Ratio = p(z|detection) / p(z|clutter)
       = (p_d * p(z|x)) / (λ * c(z))
    The constant factor (p_d / (λ * c(z))) is precomputed for efficiency since it is a constant.
    
    Args:
        measurements: Array of measurements (2, M) [range, bearing]
        track_states: Dictionary of track states {id: state_array}
        sensor_position: Sensor position [x, y]
        params: Tracking parameters
    
    Returns:
        v_factors: Array of shape (M+1, N, P) where:
            M = number of measurements
            N = number of tracks
            P = number of particles
            v_factors[0,:,:] = non-detection factors
            v_factors[1:,:,:] = detection factors
    """
    num_measurements = measurements.shape[1]
    num_tracks = len(track_states)
    
    # Initialize v-factors
    v_factors = np.zeros((num_measurements + 1, num_tracks, params.num_particles))
    
    # Set non-detection factors (probability of missed detection)
    v_factors[0, :, :] = 1 - params.detection_probability
    
    # Calculate normalization constant for likelihood ratio
    # This comes from the ratio of detection and clutter models:
    # p_d / (lambda * c(z))
    constant_factor = (1 / (2 * np.pi * np.sqrt(params.measurement_variance_bearing * 
                                               params.measurement_variance_range)) * 
                      params.detection_probability / 
                      (params.mean_clutter * params.clutter_distribution))
    
    # Compute factors for each track and measurement
    for track_idx, (track_id, track_state) in enumerate(track_states.items()):
        # Convert track state to measurement space (for all particles)
        dx = track_state[0, :] - sensor_position[0]  # x difference
        dy = track_state[1, :] - sensor_position[1]  # y difference
        
        # Calculate predicted measurements
        predicted_range = np.sqrt(dx**2 + dy**2)
        predicted_bearing = np.rad2deg(np.arctan2(dx, dy))
        
        # Compute likelihood for each measurement
        for meas_idx in range(num_measurements):
            measurement = measurements[:, meas_idx]
            
            # Range error term: exp(-0.5 * (r - r_pred)^2 / σ_r^2)
            range_error = measurement[0] - predicted_range
            range_likelihood = np.exp(-0.5 * range_error**2 / 
                                    params.measurement_variance_range)
            
            # Bearing error term: exp(-0.5 * (θ - θ_pred)^2 / σ_θ^2)
            # Note: wrap bearing error to [-180, 180] degrees
            bearing_error = wrap_to_180(measurement[1] - predicted_bearing)
            bearing_likelihood = np.exp(-0.5 * bearing_error**2 / 
                                      params.measurement_variance_bearing)
            
            # Combine likelihoods and normalize
            # v(z,x) = p(z|x) * p_d / (λ * c(z))
            v_factors[meas_idx + 1, track_idx, :] = (constant_factor * 
                                                    range_likelihood * 
                                                    bearing_likelihood)
    
    return v_factors

def perform_data_association_bp(measurements: np.ndarray,
                              track_states: Dict[int, np.ndarray],
                              sensor_position: np.ndarray,
                              params: TrackingParams) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Perform belief propagation for data association and state update.
    
    The BP algorithm solves the data association by iteratively passing
    messages between measurement and target nodes in a factor graph.
    
    Args:
        measurements: Array of measurements (2, M)
        track_states: Dictionary of track states
        sensor_position: Sensor position
        params: Tracking parameters
    
    Returns:
        association_probs: Association probabilities matrix
        updated_tracks: Updated track states
    """
    # Step 1: Compute measurement factors
    v_factors = compute_measurement_factors(measurements, track_states, 
                                         sensor_position, params)
    
    num_measurements = measurements.shape[1]
    num_tracks = len(track_states)
    
    # Step 2: Initialize messages
    # m(z→x): messages from measurements to tracks
    messages_meas_to_track = np.ones((num_measurements, num_tracks))
    
    # Perform BP iterations
    for iteration in range(params.max_iterations):
        messages_old = deepcopy(messages_meas_to_track)
        
        # Average factors over particles to get mean factor values
        mean_factors = v_factors[1:, :, :].mean(axis=2)
        
        # Update messages from measurements to tracks
        # product = m(z→x) * v(z,x)
        product = messages_meas_to_track * mean_factors
        
        # sum1 = v(∅,x) + Σ_z m(z→x)v(z,x)
        sum_product = v_factors[0, :, :].mean(axis=1) + np.sum(product, axis=0)
        
        # New message: m(z→x) = v(z,x) / (sum1 - m(z→x)v(z,x))
        messages_meas_to_track = mean_factors / (sum_product[None, :] - product)
        
        # Check convergence
        if (iteration + 1) % params.check_convergence == 0:
            delta = np.max(np.abs(np.log(messages_meas_to_track + 1e-10) - 
                                np.log(messages_old + 1e-10)))
            if delta < params.convergence_threshold:
                break
    
    # Compute final association probabilities
    association_probs = messages_meas_to_track * mean_factors
    association_probs = association_probs / (np.sum(association_probs, axis=1, 
                                                  keepdims=True) + 1e-10)
    
    # Update track states based on associations
    updated_tracks = {}
    for track_idx, (track_id, track_state) in enumerate(track_states.items()):
        # Only update tracks with significant measurement associations
        track_assoc_probs = association_probs[:, track_idx]
        if np.max(track_assoc_probs) > params.existence_threshold:
            updated_state = update_track_state(track_state, measurements, 
                                            track_assoc_probs, sensor_position, params)
            updated_tracks[track_id] = updated_state
    
    return association_probs, updated_tracks

def update_track_state(track_state: np.ndarray,
                      measurements: np.ndarray,
                      association_probs: np.ndarray,
                      sensor_position: np.ndarray,
                      params: TrackingParams) -> np.ndarray:
    """
    Update track state using soft measurement assignment.
    
    Performs a weighted update of the particle states based on
    association probabilities with each measurement.
    """
    updated_state = track_state.copy()
    
    for meas_idx, prob in enumerate(association_probs):
        if prob > 0.01:  # Only update for significant associations
            measurement = measurements[:, meas_idx]
            
            # For each particle
            for p in range(params.num_particles):
                # Current particle state
                dx = updated_state[0, p] - sensor_position[0]
                dy = updated_state[1, p] - sensor_position[1]
                
                # Predicted measurement
                pred_range = np.sqrt(dx**2 + dy**2)
                pred_bearing = np.rad2deg(np.arctan2(dx, dy))
                
                # Innovation (measurement residual)
                range_innov = measurement[0] - pred_range
                bearing_innov = wrap_to_180(measurement[1] - pred_bearing)
                
                # Convert polar innovation to Cartesian
                bearing_rad = np.deg2rad(pred_bearing)
                dx_update = (range_innov * np.sin(bearing_rad) - 
                           pred_range * bearing_innov * np.cos(bearing_rad))
                dy_update = (range_innov * np.cos(bearing_rad) + 
                           pred_range * bearing_innov * np.sin(bearing_rad))
                
                # Apply weighted update
                updated_state[0, p] += prob * dx_update
                updated_state[1, p] += prob * dy_update
    
    # Add process noise
    updated_state += np.random.normal(0, params.process_noise_std, 
                                    updated_state.shape)
    
    return updated_state

def wrap_to_180(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180] degree range."""
    return (angle + 180) % 360 - 180