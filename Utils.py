import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import numpy.typing as npt


def get_sensor_positions(num_sensors: int, radius: float) -> np.ndarray:
    """Generate sensor positions in a circle."""
    num_sensors = round(num_sensors)
    if num_sensors < 2:
        return np.array([[0], [radius]])
    
    sensor_positions = np.zeros((2, num_sensors))
    sensor_positions[:, 0] = [0, radius]
    step_size = 2 * np.pi / num_sensors
    
    angle = 0
    for sensor in range(1, num_sensors):
        angle += step_size
        sensor_positions[:, sensor] = [np.sin(angle) * radius, np.cos(angle) * radius]
    
    return sensor_positions

def get_start_states(num_targets: int, radius: float, velocity: float) -> np.ndarray:
    """Generate initial states for targets."""
    states = np.zeros((4, num_targets))
    angles = np.linspace(0, 2*np.pi, num_targets+1)[:-1]
    
    for i in range(num_targets):
        states[0:2, i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
        states[2:4, i] = [-velocity * np.sin(angles[i]), velocity * np.cos(angles[i])]
    
    return states

def get_transition_matrices(scan_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get state transition and noise matrices."""
    A = np.eye(4)
    A[0, 2] = scan_time
    A[1, 3] = scan_time
    
    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time**2
    W[1, 1] = 0.5 * scan_time**2
    W[2, 0] = scan_time
    W[3, 1] = scan_time
    
    return A, W

def generate_true_tracks(parameters: Parameters, num_steps: int) -> np.ndarray:
    """Generate true target tracks."""
    num_targets = parameters.target_start_states.shape[1]
    target_tracks = np.full((4, num_targets, num_steps), np.nan)
    
    for target in range(num_targets):
        current_state = parameters.target_start_states[:, target]
        
        for step in range(1, num_steps):
            A, W = get_transition_matrices(parameters.length_steps[step])
            current_state = A @ current_state + W @ np.sqrt(parameters.driving_noise_variance) * np.random.randn(2)
            
            if (parameters.target_appearance_from_to[0, target] <= step and 
                step <= parameters.target_appearance_from_to[1, target]):
                target_tracks[:, target, step] = current_state
    
    return target_tracks

def generate_measurements(target_trajectory: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Generate measurements for all sensors."""
    num_targets = target_trajectory.shape[1]
    measurements = np.zeros((2, num_targets, parameters.num_sensors))
    
    for sensor in range(parameters.num_sensors):
        for target in range(num_targets):
            dx = target_trajectory[0, target] - parameters.sensor_positions[0, sensor]
            dy = target_trajectory[1, target] - parameters.sensor_positions[1, sensor]
            
            range_measurement = np.sqrt(dx**2 + dy**2) + np.sqrt(parameters.measurement_variance_range) * np.random.randn()
            bearing_measurement = np.degrees(np.arctan2(dx, dy)) + np.sqrt(parameters.measurement_variance_bearing) * np.random.randn()
            
            measurements[:, target, sensor] = [range_measurement, bearing_measurement]
    
    return measurements

def generate_cluttered_measurements(track_measurements: np.ndarray, parameters: Parameters) -> List[np.ndarray]:
    """Generate cluttered measurements for all sensors."""
    all_measurements = []
    
    for sensor in range(parameters.num_sensors):
        # Determine detections
        does_exist = ~np.isnan(track_measurements[0, :, sensor])
        is_detected = np.random.rand(parameters.target_start_states.shape[1]) < parameters.detection_probability
        detected_measurements = track_measurements[:, does_exist & is_detected, sensor]
        
        # Generate clutter
        num_clutter = np.random.poisson(parameters.mean_clutter)
        clutter = np.zeros((2, num_clutter))
        clutter[0, :] = parameters.measurement_range * np.random.rand(num_clutter)
        clutter[1, :] = 360 * np.random.rand(num_clutter) - 180
        
        # Combine measurements and randomize
        all_meas = np.hstack([clutter, detected_measurements])
        np.random.shuffle(all_meas.T)
        
        all_measurements.append(all_meas)
    
    return all_measurements

class Visualizer:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the plot with surveillance region and sensor positions."""
        sr = self.parameters.surveillance_region
        self.ax.set_xlim(sr[0])
        self.ax.set_ylim(sr[1])
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Plot sensor positions
        self.ax.scatter(self.parameters.sensor_positions[0, :],
                       self.parameters.sensor_positions[1, :],
                       c='blue', marker='^', s=100, label='Sensors')
        
    def update(self, true_tracks: np.ndarray, measurements: List[np.ndarray], 
               current_step: int):
        """Update the plot with current tracks and measurements."""
        self.ax.clear()
        self.setup_plot()
        
        # Plot true trajectories (up to current step)
        for target in range(true_tracks.shape[1]):
            valid_track = ~np.isnan(true_tracks[0, target, :current_step+1])
            if np.any(valid_track):
                self.ax.plot(true_tracks[0, target, :current_step+1][valid_track],
                           true_tracks[1, target, :current_step+1][valid_track],
                           'r-', linewidth=1, alpha=0.5)
        
        # Plot current measurements
        for sensor_meas in measurements:
            if sensor_meas.size > 0:
                # Convert polar to Cartesian for visualization
                for sensor in range(self.parameters.num_sensors):
                    sp = self.parameters.sensor_positions[:, sensor]
                    ranges = sensor_meas[0, :]
                    bearings = np.radians(sensor_meas[1, :])
                    x = sp[0] + ranges * np.sin(bearings)
                    y = sp[1] + ranges * np.cos(bearings)
                    self.ax.scatter(x, y, c='black', s=10, alpha=0.5)
        
        plt.pause(0.1)
