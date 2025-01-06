import numpy as np
from scipy.linalg import sqrtm
from typing import Dict, Tuple, Optional

class BP_Particle_Filter:
    """
    Implements the vector-type system model for MTT with unknown, time-varying number of targets
    as described in Section VIII of Meyer et al.
    """
    
    def __init__(self, parameters: Dict):
        """
        Initialize the MTT system with given parameters.
        
        Args:
            parameters: Dictionary containing:
                - mu_n: mean number of newly detected targets (μ_n^(s))
                - mu_c: mean number of clutter measurements (μ_c^(s))
                - p_d: detection probability (p_d^(s))
                - p_s: survival probability (p_s)
        """
        self.mu_n = parameters['mu_n']
        self.mu_c = parameters['mu_c']
        self.p_d = parameters['p_d']
        self.p_s = parameters['p_s']
        self.dt = parameters['dt']
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])   # State transition matrix
        self.W = np.array([[0.5*self.dt**2, 0], 
                            [0, 0.5*self.dt**2], 
                            [self.dt, 0], 
                            [0, self.dt]])  # Process noise matrix
        self.particles = 
        self.state = np.zeros(4)  # Initial state
        self.j_k = 0  # Total number of PTs at time k (initialized to 0 per Vu7)

    def _predict_state(self) -> np.ndarray:

 
        
    def state_transition(self, y_k_prev: np.ndarray, r_k_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements the state transition model from equations (53)-(55).
        
        For each PT j at k-1:
        - If r_k_prev[j] = 0 (not existing), then r_k[j] = 0 (still not existing)
        - If r_k_prev[j] = 1 (existing), then:
            * r_k[j] = 1 with probability p_s (survives)
            * r_k[j] = 0 with probability 1-p_s (disappears)
            
        Args:
            y_k_prev: Previous states [y_k_prev^(1),...,y_k_prev^(j_k_prev)]
            r_k_prev: Previous existence variables [r_k_prev^(1),...,r_k_prev^(j_k_prev)]
            
        Returns:
            y_k: Updated states
            r_k: Updated existence variables
        """
        j_k_prev = len(y_k_prev)
        y_k = np.zeros_like(y_k_prev)
        r_k = np.zeros_like(r_k_prev)
        
        for j in range(j_k_prev):
            if r_k_prev[j] == 0:
                # Implement equation (54): non-existing targets stay non-existing
                r_k[j] = 0
                y_k[j] = self._sample_dummy_pdf()
            else:
                # Implement equation (55): existing targets survive with p_s
                survives = np.random.rand() < self.params['p_s']
                r_k[j] = 1 if survives else 0
                if survives:
                    y_k[j] = self._state_transition_pdf(y_k_prev[j])
                else:
                    y_k[j] = self._sample_dummy_pdf()
                    
        return y_k, r_k
    
    def measurement_evaluation(self, z_k_s: np.ndarray, y_legacy: np.ndarray, 
                             r_legacy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Implements measurement evaluation for both legacy and new PTs according to 
        equations (56)-(58).
        
        Args:
            z_k_s: Measurements at time k from sensor s
            y_legacy: Legacy PT states y_k_s
            r_legacy: Legacy PT existence variables r_k_s
            
        Returns:
            a_k_s: Data association vector
            r_new: Existence variables for new PTs
            y_new: States for new PTs
        """
        m_k_s = len(z_k_s)  # Number of measurements
        j_k_s = len(y_legacy)  # Number of legacy PTs
        
        # Calculate association probabilities according to equation (57)
        association_probs = self._calculate_association_probs(
            z_k_s, y_legacy, r_legacy, m_k_s, j_k_s)
        
        # Sample associations based on equation (56)
        a_k_s = self._sample_associations(association_probs, m_k_s, j_k_s)
        
        # Initialize new PTs
        r_new = np.zeros(m_k_s)
        y_new = np.zeros((m_k_s, y_legacy.shape[1]))
        
        # Process each measurement to determine if it's from a new target
        for m in range(m_k_s):
            if not any(a_k_s == m):  # No legacy PT associated
                # Probability of new target vs clutter according to equation (56)
                prob_new = self.params['mu_n'] / (self.params['mu_n'] + self.params['mu_c'])
                if np.random.rand() < prob_new:
                    r_new[m] = 1
                    y_new[m] = self._sample_new_target_state(z_k_s[m])
                    
        return a_k_s, r_new, y_new
    
    def _calculate_association_probs(self, z_k_s: np.ndarray, y_legacy: np.ndarray, 
                                  r_legacy: np.ndarray, m_k_s: int, j_k_s: int) -> np.ndarray:
        """
        Implements the calculation of q_1 terms from equation (58).
        """
        probs = np.zeros((j_k_s, m_k_s + 1))  # +1 for no detection case
        
        for j in range(j_k_s):
            if r_legacy[j]:
                # Detection probability terms
                p_d = self.params['p_d']
                probs[j, 1:] = p_d / self.params['mu_c']  # For measurement associations
                probs[j, 0] = 1 - p_d  # For no detection
                
        return probs
    
    def _state_transition_pdf(self, y_prev: np.ndarray) -> np.ndarray:
        """Sample from state transition PDF f(x_k|x_k-1)."""
        # Implementation depends on specific motion model
        pass
    
    def _sample_dummy_pdf(self) -> np.ndarray:
        """Sample from dummy PDF f_D(x) for non-existing targets."""
        # Implementation depends on specific state space
        pass
    
    def _sample_new_target_state(self, z: np.ndarray) -> np.ndarray:
        """Sample new target state based on measurement."""
        # Implementation depends on specific measurement model
        pass

    def state_transition(self, prev_targets: List[Target], dt: float) -> List[Target]:
        """
        Implements the state transition model from Eq. (53)-(55).
        f(y_k|y_{k-1}) = ∏_{j=1}^{j_{k-1}} f(y_k^(j)|y_{k-1}^(j))
        """
        new_targets = []
        
        # Process each target independently (Eq. 53)
        for target in prev_targets:
            if not target.exists:
                # Eq. 54: If target didn't exist, it still doesn't exist
                new_target = Target(
                    state=self._dummy_state(),
                    exists=False,
                    id=target.id
                )
            else:
                # Eq. 55: Target may survive with probability p_s
                survives = np.random.random() < self.p_s
                if survives:
                    # State transition for surviving target
                    new_state = self._predict_state(target.state, dt)
                    new_target = Target(state=new_state, exists=True, id=target.id)
                else:
                    new_target = Target(
                        state=self._dummy_state(),
                        exists=False,
                        id=target.id
                    )
            new_targets.append(new_target)
            
        return new_targets

    def measurement_update(self, 
                          targets: List[Target], 
                          measurements: np.ndarray) -> Tuple[List[Target], np.ndarray]:
        """
        Implements measurement update using Eq. (56) for data association probability
        and Eq. (63) for measurement likelihood.
        """
        m_k = len(measurements)  # Number of measurements
        j_k = len(targets)       # Number of potential targets
        
        # Calculate data association probabilities (Eq. 56)
        # p(a_k,s, r̄_k,s, m_k,s|y̲_k,s)
        association_probs = np.zeros((j_k + 1, m_k))  # +1 for new target possibility
        
        for j in range(j_k):
            if targets[j].exists:
                # Probability of detection term from Eq. 56
                detection_prob = self.p_d
                
                for m in range(m_k):
                    # Measurement likelihood term from Eq. 63
                    likelihood = self._measurement_likelihood(measurements[m], targets[j].state)
                    association_probs[j, m] = detection_prob * likelihood / self.mu_c
        
        # Add probabilities for new targets (last row)
        for m in range(m_k):
            association_probs[-1, m] = self.mu_n / self.mu_c
            
        return self._update_states(targets, measurements, association_probs)

    def _predict_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Simple constant velocity prediction."""
        F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
        return F @ state

    def _measurement_likelihood(self, 
                              measurement: np.ndarray, 
                              state: np.ndarray, 
                              R: float = 1.0) -> float:
        """
        Implements measurement likelihood f^(s)(z|x) from Eq. (63).
        Assumes position-only measurements with Gaussian noise.
        """
        H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement matrix for position-only
        predicted_meas = H @ state
        innovation = measurement - predicted_meas
        
        # Gaussian likelihood
        return stats.multivariate_normal.pdf(innovation, mean=np.zeros(2), cov=R*np.eye(2))

    def _dummy_state(self) -> np.ndarray:
        """Returns a dummy state for non-existent targets (used in Eq. 51)."""
        return np.zeros(4)

    def _update_states(self, 
                      targets: List[Target], 
                      measurements: np.ndarray,
                      association_probs: np.ndarray) -> Tuple[List[Target], np.ndarray]:
        """
        Updates target states based on measurement associations.
        Implements state update part of Eq. (67).
        """
        updated_targets = targets.copy()
        # Use Hungarian algorithm or other assignment method here
        # For simplicity, we'll use greedy assignment
        assignments = self._greedy_assignment(association_probs)
        
        # Update existing targets and add new ones
        for m, j in enumerate(assignments):
            if j == len(targets):  # New target
                new_state = np.array([measurements[m][0], measurements[m][1], 0, 0])
                updated_targets.append(Target(state=new_state, exists=True, 
                                           id=len(updated_targets)))
            elif j >= 0:  # Update existing target
                # Simple Kalman update would go here
                updated_targets[j].state[:2] = measurements[m]
                
        return updated_targets, assignments

    def _greedy_assignment(self, association_probs: np.ndarray) -> np.ndarray:
        """Simple greedy measurement-to-target assignment."""
        m_k = association_probs.shape[1]
        assignments = -np.ones(m_k, dtype=int)
        
        for m in range(m_k):
            j = np.argmax(association_probs[:, m])
            assignments[m] = j
            
        return assignments
    

def introduce_new_pts(new_measurements: np.ndarray,
                     sensor: int,
                     step: int,
                     unknown_number: float,
                     unknown_particles: np.ndarray,
                     parameters: Parameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements introduceNewPTs.m for introducing new potential targets.
    
    Args:
        new_measurements: New measurements matrix
        sensor: Current sensor index
        step: Current time step
        unknown_number: Number of unknown targets
        unknown_particles: Particle representation of unknown targets
        parameters: System parameters
    
    Returns:
        new_pts: New potential targets
        new_labels: Labels for new targets
        new_existences: Existence probabilities for new targets
        xi_messages: Association messages
    """
    num_measurements = new_measurements.shape[1]
    
    # Calculate unknown intensity
    surveillance_area = ((parameters.surveillance_region[1] - parameters.surveillance_region[0]) * 
                        (parameters.surveillance_region[3] - parameters.surveillance_region[2]))
    unknown_intensity = unknown_number / surveillance_area
    unknown_intensity *= (1 - parameters.detection_probability) ** (sensor - 1)
    
    clutter_intensity = parameters.mean_clutter * (1.0 / surveillance_area)
    
    # Initialize outputs
    new_pts = np.zeros((4, parameters.num_particles, num_measurements))
    new_labels = np.zeros((3, num_measurements))
    xi_messages = np.zeros(num_measurements)
    
    if num_measurements > 0:
        constants = calculate_constants_uniform(
            parameters.sensor_positions[:, sensor-1],
            new_measurements,
            unknown_particles,
            parameters
        )
        
        for m in range(num_measurements):
            new_pts[:, :, m] = sample_from_likelihood(
                new_measurements[:, m],
                sensor,
                parameters.num_particles,
                parameters
            )
            new_labels[:, m] = [step, sensor, m + 1]
            xi_messages[m] = 1 + (constants[m] * unknown_intensity * 
                                parameters.detection_probability) / clutter_intensity
    
    new_existences = xi_messages - 1
    
    return new_pts, new_labels, new_existences, xi_messages

def perform_prediction(old_particles: np.ndarray,
                      old_existences: np.ndarray,
                      scan_time: float,
                      parameters: Parameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements performPrediction.m for state prediction.
    
    Args:
        old_particles: Previous particle states
        old_existences: Previous existence probabilities
        scan_time: Time between scans
        parameters: System parameters
    
    Returns:
        new_particles: Predicted particle states
        new_existences: Predicted existence probabilities
    """
    num_particles = old_particles.shape[1]
    num_targets = old_particles.shape[2]
    
    # Get transition matrices
    A, W = get_transition_matrices(scan_time)
    
    new_particles = old_particles.copy()
    new_existences = old_existences.copy()
    
    for target in range(num_targets):
        # State prediction
        driving_noise = np.random.randn(2, num_particles)
        new_particles[:, :, target] = (A @ old_particles[:, :, target] + 
                                     W @ (np.sqrt(parameters.driving_noise_variance) * driving_noise))
        # Existence prediction
        new_existences[target] = parameters.survival_probability * old_existences[target]
    
    return new_particles, new_existences

def get_transition_matrices(scan_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to create state transition matrices."""
    A = np.eye(4)
    A[0, 2] = scan_time
    A[1, 3] = scan_time
    
    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time**2
    W[1, 1] = 0.5 * scan_time**2
    W[2, 0] = scan_time
    W[3, 1] = scan_time
    
    return A, W

def sample_from_likelihood(measurement: np.ndarray,
                         sensor_index: int,
                         num_particles: int,
                         parameters: Parameters) -> np.ndarray:
    """
    Implements sampleFromLikelihood.m for particle generation from measurement.
    """
    sensor_pos = parameters.sensor_positions[:, sensor_index-1]
    
    samples = np.zeros((4, num_particles))
    
    # Generate random range and bearing
    random_range = (measurement[0] + 
                   np.sqrt(parameters.measurement_variance_range) * 
                   np.random.randn(num_particles))
    random_bearing = (measurement[1] + 
                     np.sqrt(parameters.measurement_variance_bearing) * 
                     np.random.randn(num_particles))
    
    # Convert to Cartesian coordinates
    samples[0] = sensor_pos[0] + random_range * np.sin(np.deg2rad(random_bearing))
    samples[1] = sensor_pos[1] + random_range * np.cos(np.deg2rad(random_bearing))
    
    # Sample velocities
    velocity_samples = np.random.randn(2, num_particles)
    vel_cov_sqrt = np.linalg.cholesky(parameters.prior_velocity_covariance)
    samples[2:4] = vel_cov_sqrt @ velocity_samples
    
    return samples