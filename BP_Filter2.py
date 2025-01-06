import numpy as np
from scipy.stats import multivariate_normal
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt

@dataclass
class Parameters:
    """System parameters for MTT."""
    detection_probability: float
    survival_probability: float
    mean_clutter: float
    measurement_variance_range: float
    measurement_variance_bearing: float
    driving_noise_variance: float
    num_particles: int
    prior_velocity_covariance: np.ndarray
    surveillance_region: np.ndarray  # [x_min, x_max, y_min, y_max]
    sensor_positions: np.ndarray     # Shape: (2, num_sensors)

@dataclass
class Target:
    """Represents a potential target with state and existence probability."""
    particles: np.ndarray  # Shape: (4, num_particles)
    existence_prob: float
    label: np.ndarray      # [time_step, sensor_id, measurement_id]
    id: int

class MTTFilter:
    """Multi-Target Tracking filter using belief propagation."""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.targets: List[Target] = []
        self.next_id = 0
    
    def predict(self, dt: float) -> None:
        """Predict step for all targets."""
        A, W = self._get_transition_matrices(dt)
        
        for target in self.targets:
            # State prediction for particles
            driving_noise = np.random.randn(2, self.params.num_particles)
            target.particles = (A @ target.particles + 
                              W @ (np.sqrt(self.params.driving_noise_variance) * driving_noise))
            
            # Update existence probability
            target.existence_prob *= self.params.survival_probability
    
    def update(self, measurements: np.ndarray, sensor_idx: int, time_step: int) -> None:
        """Update step using current measurements."""
        if len(measurements) == 0:
            return
        
        # Data association using BP
        association_probs = self._perform_data_association(measurements)
        
        # Create new targets and update existing ones
        self._process_measurements(measurements, association_probs, sensor_idx, time_step)
        
        # Remove targets with low existence probability
        self._prune_targets()
    
    def _get_transition_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create state transition matrices."""
        A = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
        
        W = np.array([[0.5*dt**2, 0],
                     [0, 0.5*dt**2],
                     [dt, 0],
                     [0, dt]])
        
        return A, W
    
    def _perform_data_association(self, measurements: np.ndarray) -> np.ndarray:
        """Perform belief propagation for data association."""
        num_measurements = len(measurements)
        num_targets = len(self.targets)
        
        if num_targets == 0:
            return np.ones((num_measurements, 1)) * self.params.detection_probability
        
        # Initialize association probabilities
        association_probs = np.zeros((num_measurements, num_targets + 1))  # +1 for new target
        
        # Calculate measurement likelihoods
        for j, target in enumerate(self.targets):
            if target.existence_prob > 0:
                for m in range(num_measurements):
                    likelihood = self._measurement_likelihood(measurements[m], target.particles)
                    association_probs[m, j] = (likelihood * self.params.detection_probability / 
                                             self.params.mean_clutter)
        
        # New target probabilities
        association_probs[:, -1] = self.params.detection_probability
        
        return association_probs
    
    def _measurement_likelihood(self, measurement: np.ndarray, particles: np.ndarray) -> float:
        """Calculate measurement likelihood for particles."""
        sensor_pos = self.params.sensor_positions[:, 0]  # Using first sensor for simplicity
        
        # Convert particles to polar coordinates
        dx = particles[0, :] - sensor_pos[0]
        dy = particles[1, :] - sensor_pos[1]
        pred_range = np.sqrt(dx**2 + dy**2)
        pred_bearing = np.arctan2(dy, dx)
        
        # Calculate likelihood
        range_likelihood = multivariate_normal.pdf(
            measurement[0], pred_range.mean(), self.params.measurement_variance_range)
        bearing_likelihood = multivariate_normal.pdf(
            measurement[1], pred_bearing.mean(), self.params.measurement_variance_bearing)
        
        return range_likelihood * bearing_likelihood
    
    def _process_measurements(self, measurements: np.ndarray, association_probs: np.ndarray, 
                            sensor_idx: int, time_step: int) -> None:
        """Process measurements using association probabilities."""
        assignments = self._assign_measurements(association_probs)
        
        for m, j in enumerate(assignments):
            if j == association_probs.shape[1] - 1:  # New target
                self._create_new_target(measurements[m], sensor_idx, time_step, m)
            elif j >= 0:  # Update existing target
                self._update_target(self.targets[j], measurements[m])
    
    def _assign_measurements(self, association_probs: np.ndarray) -> np.ndarray:
        """Simple greedy measurement assignment."""
        assignments = np.argmax(association_probs, axis=1)
        return assignments
    
    def _create_new_target(self, measurement: np.ndarray, sensor_idx: int, 
                          time_step: int, meas_idx: int) -> None:
        """Create new target from measurement."""
        particles = self._sample_from_measurement(measurement)
        label = np.array([time_step, sensor_idx, meas_idx])
        
        new_target = Target(
            particles=particles,
            existence_prob=0.8,  # Initial existence probability
            label=label,
            id=self.next_id
        )
        self.next_id += 1
        self.targets.append(new_target)
    
    def _sample_from_measurement(self, measurement: np.ndarray) -> np.ndarray:
        """Generate particles from measurement."""
        particles = np.zeros((4, self.params.num_particles))
        sensor_pos = self.params.sensor_positions[:, 0]  # Using first sensor for simplicity
        
        # Generate random range and bearing
        random_range = (measurement[0] + np.sqrt(self.params.measurement_variance_range) * 
                       np.random.randn(self.params.num_particles))
        random_bearing = (measurement[1] + np.sqrt(self.params.measurement_variance_bearing) * 
                         np.random.randn(self.params.num_particles))
        
        # Convert to Cartesian coordinates
        particles[0, :] = sensor_pos[0] + random_range * np.cos(random_bearing)
        particles[1, :] = sensor_pos[1] + random_range * np.sin(random_bearing)
        
        # Sample velocities
        vel_cov_sqrt = np.linalg.cholesky(self.params.prior_velocity_covariance)
        particles[2:4, :] = vel_cov_sqrt @ np.random.randn(2, self.params.num_particles)
        
        return particles
    
    def _update_target(self, target: Target, measurement: np.ndarray) -> None:
        """Update target state using measurement."""
        weights = self._measurement_likelihood(measurement, target.particles)
        weights /= np.sum(weights)
        
        # Resample particles
        indices = np.random.choice(self.params.num_particles, size=self.params.num_particles, 
                                 p=weights)
        target.particles = target.particles[:, indices]
    
    def _prune_targets(self, threshold: float = 0.1) -> None:
        """Remove targets with low existence probability."""
        self.targets = [target for target in self.targets 
                       if target.existence_prob > threshold]