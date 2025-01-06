import numpy as np
from scipy.linalg import sqrtm
from typing import Dict, Tuple, Optional
from copy import deepcopy

class BP_Particle_Filter:
    """
    Implements the vector-type system model for MTT with unknown, time-varying number of targets
    as described in Section VIII of Meyer et al.

    Notice this implementation has all the functions in one place,
    this is mostly from debugging purposes and to make it easier to follow the code.
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
        self.measurement_variance_bearing = parameters['measurement_variance_bearing']
        self.measurement_variance_range = parameters['measurement_variance_range']
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])   # State transition matrix
        self.W = np.array([[0.5*self.dt**2, 0], 
                            [0, 0.5*self.dt**2], 
                            [self.dt, 0], 
                            [0, self.dt]])  # Process noise matrix
        self.numParticles = parameters['numParticles']
        self.steps = parameters['steps']
        self.numTargets = 0 # TODO THIS NEED TO BE CHANGED
        self.posterior_belief = np.zeros((4, self.numParticles, 0))
        self.existence_prob = np.zeros((0, 1))
        self.estimated_cardinality = np.zeros(parameters['steps'])
        self.estimated_states = np.zeros((4, parameters['steps']))
        # Initialize Poisson point process
        self.unknown_number = parameters['unknownNumber']
        self.unknown_particles = np.zeros((2, parameters['numParticles']))
        # Initialize arrays for new PTs and messages
        self.new_pts = np.zeros((4, self.num_particles, num_measurements))
        self.new_labels = np.zeros((3, num_measurements))
        self.xi_k = np.zeros(num_measurements)
        # Initiate the messages for the Loopy BP
        self.v_k_current = np.ones((num_measurements, num_objects))  # v_k^[l](m→i)

        # Loop through x and y coordinates (i=0 for x, i=1 for y)
        for i in range(2):
            # Generate uniform random distribution of particles within surveillance region:
            # 1. Calculate region width: surveillanceRegion[1,i] - surveillanceRegion[0,i]
            # 2. Generate random values [0,1]: np.random.rand(numParticles)
            # 3. Scale random values to region width by multiplication
            # 4. Shift by region minimum (surveillanceRegion[0,i]) to get final positions
            self.unknown_particles[i,:] = (parameters['surveillanceRegion'][1,i] - 
                                 parameters['surveillanceRegion'][0,i]) * \
                                 np.random.rand(parameters['numParticles']) + \
                                 parameters['surveillanceRegion'][0,i]
        self.j_k = 0  # Total number of PTs at time k (initialized to 0 per Vu7)

    def prediction(self) -> np.ndarray:
        for target in range(self.numTargets):
            # State prediction
            driving_noise = np.random.randn(2, self.numParticles)
            self.posterior_belief[:, :, target] = (self.F @ self.posterior_belief[:, :, target] + 
                                               self.W @ (np.sqrt(self.driving_noise_variance) * driving_noise))
            # Existence prediction
            self.existence_prob[target] = self.survival_probability * self.existence_prob[target]

    def compute_alpha_messages(self):
        """
        Compute alpha messages using Sequential Importance Resampling (SIR) particle filter
        """
        num_targets = prev_states.shape[2]
        state_dim = prev_states.shape[0]
        
        # Initialize output arrays
        alpha_states = np.zeros((state_dim, self.params.num_particles, num_targets))
        alpha_weights = np.zeros((self.params.num_particles, num_targets))
        alpha_existence = np.zeros(num_targets)

        # Process existing targets
        for i in range(prev_states.shape[2]):
            if prev_existence[i] > 0:
                # Predict existence probability using survival probability
                # p(rk(i)=1 | rk-1(i)=1) = ps(xk-1(i)) from eq (55)
                alpha_existence[i] = prev_existence[i] * self.params.survival_prob

                # Predict particles through state transition model
                # This implements f(xk(i)|xk-1(i)) from eq (25)
                for p in range(self.params.num_particles):
                    # Sample process noise
                    noise = np.random.multivariate_normal(
                        np.zeros(state_dim), 
                        self.Q
                    )
                    
                    # State prediction
                    alpha_states[:,p,i] = (
                        self.F @ prev_states[:,p,i] + noise
                    )
                    
                    # Weights remain unchanged in prediction step
                    alpha_weights[p,i] = prev_weights[p,i]

                # Check if resampling is needed
                if self._need_resampling(alpha_weights[:,i]):
                    alpha_states[:,:,i], alpha_weights[:,i] = self._resample(
                        alpha_states[:,:,i],
                        alpha_weights[:,i]
                    )

    def introduce_new_pts(self, measurements):
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
        num_measurements = measurements.shape[1]
    
        # Compute unknown intensity (corresponds to intensity of undetected targets)
        # This affects the messages passed through factors q^m in the factor graph
        surveillance_area = ((parameters['surveillance_region'][0, 0] - parameters['surveillance_region'][1, 0]) * 
                            (parameters['surveillance_region'][0, 1] - parameters['surveillance_region'][1, 1]))
        unknown_intensity = unknown_number / surveillance_area
        # This part need more comments
        unknown_intensity *= (1 - self.p_d) ** (self.sensor - 1)
    
        # Calculate constants if there are measurements
        if num_measurements:
            constants = self.calculate_constants_uniform()
    
        # Initialize arrays for new PTs and messages
        self.new_pts = np.zeros((4, self.num_particles, num_measurements))
        self.new_labels = np.zeros((3, num_measurements))
        self.xi_k = np.zeros(num_measurements)
    
        # Process each measurement to create new PTs
        # This corresponds to creating new nodes a^m and b^m in the factor graph
        for measurement in range(num_measurements):
            # Sample new particles based on measurement likelihood
            self.new_pts[:, :, measurement] = self.sample_from_likelihood(measurements[:, measurement])
            
            # Assign labels [step; sensor; measurement]
            self.new_labels[:, measurement] = [self.step, self.sensor, measurement]
            
            # Compute xi messages (ξ_m in the factor graph)
            # These messages flow from measurement factors to existence variables
            self.xi_k[measurement] = 1 + (constants[measurement] * unknown_intensity * 
                                          self.p_d) / self.c
    
        # Compute existence probabilities for new targets
        self.new_existences = self.xi_k - 1
    
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
    
    def evaluate_measurements(self, measurements, sensor_position):
        """
        Evaluate measurement likelihood messages according to equations (63)-(66)
        from the paper.
        
        Returns:
            beta_messages: Messages from measurement nodes to target nodes (eq. 64)
            v_factors1: Measurement evaluation factors for existing targets (eq. 66)
        """
        num_measurements = measurements.shape[1]
        num_targets = self.numTargets
        
        # Initialize messages and factors
        # beta_messages shape: (num_measurements+1 x num_targets)
        # Extra row for non-detection case
        self.beta_k = np.zeros((num_measurements + 1, num_targets))
        
        # v_factors1 shape: (num_measurements+1 x num_targets x num_particles)
        # Stores likelihood factors for each particle
        self.v_factors1 = np.zeros((num_measurements + 1, num_targets, self.num_particles))
        
        if num_targets == 0:
            #NEED TO BE CHANGED
            pass
            
        # Calculate non-detection likelihood (v_factors1[0,:,:])
        # This corresponds to eq. 66 for r̅_k,s^(m) = 0
        self.v_factors1[0,:,:] = 1 - self.p_d
        
        # Calculate constant factor according to measurement model
        # This implements part of eq. 63-64 for the measurement likelihood
        constant_factor = (1 / (2 * np.pi * np.sqrt(self.meas_var_bearing * 
                          self.meas_var_range)) * self.p_d / 
                          (self.mean_clutter * self.clutter_distribution))
        
        
        # For each target, calculate measurement likelihoods
        for target in range(num_targets):
            # Calculate predicted range and bearing for all particles
            dx = self.alphas[0,:,target] - sensor_position[0]
            dy = self.alphas[1,:,target] - sensor_position[1]
            predicted_range = np.sqrt(dx**2 + dy**2)
            predicted_bearing = np.degrees(np.arctan2(dx, dy))
            
            # For each measurement, calculate likelihood for all particles
            # This implements eq. 65-66 for the measurement likelihood factors
            for m in range(num_measurements):
                # Range likelihood
                range_likelihood = np.exp(-0.5/self.meas_var_range * 
                                  (measurements[0,m] - predicted_range)**2)
                
                # Bearing likelihood with wrapping to [-180,180]
                bearing_diff = measurements[1,m] - predicted_bearing
                bearing_diff = ((bearing_diff + 180) % 360) - 180  # wrap to [-180,180]
                bearing_likelihood = np.exp(-0.5/self.meas_var_bearing * bearing_diff**2)
                
                # Combined likelihood for measurement m
                self.v_factors1[m+1,target,:] = constant_factor * range_likelihood * bearing_likelihood
        
        # Calculate v_factors0 for non-existent targets (eq. 65 for r̅_k,s^(m) = 0)
        v_factors0 = np.zeros((num_measurements + 1, num_targets))
        v_factors0[0,:] = 1
        
        # Combine factors according to existence probability (eq. 64)
        self.existence = np.tile(alphas_existence.T, (num_measurements + 1, 1))
        self.beta_k = (existence * np.mean(v_factors1, axis=2) + 
                        (1 - existence) * v_factors0)
        

    def perform_data_association_bp(self, beta_k):
        """
        Implements the scalable Sum-Product Algorithm (SPA) based Data Association (DA) algorithm.
        
        This implementation follows equations (30) and (31) from the paper, which represent a 
        simplified version of the message passing scheme for binary consistency constraints.

        Notice that the message for the missed detection hyptothesis should not be updated
        Therefore, in every iteration this probability is used to compute the normalization factor
        This specific consideration makes this implementation different from other BP papers
        
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
        for l in range(self.num_iterations):
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
            distance = np.max(np.abs(np.log(v_k_current/v_k_previous)))
            if distance < self.threshold:
                break
        
        phi_k = v_k_current
        return phi_k, v_k
    
     
            
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