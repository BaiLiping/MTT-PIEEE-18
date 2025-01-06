import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import numpy.typing as npt
from Utils import get_sensor_positions, get_start_states, generate_true_tracks, generate_measurements, generate_cluttered_measurements
from BP_Particle_Filter import BPFilter


@dataclass
class Parameters:
    surveillance_region: np.ndarray
    prior_velocity_covariance: np.ndarray
    driving_noise_variance: float
    length_steps: np.ndarray
    survival_probability: float
    num_sensors: int
    measurement_variance_range: float
    measurement_variance_bearing: float
    detection_probability: float
    measurement_range: float
    mean_clutter: float
    clutter_distribution: float
    detection_threshold: float
    threshold_pruning: float
    minimum_track_length: int
    num_particles: int
    target_start_states: np.ndarray
    target_appearance_from_to: np.ndarray
    sensor_positions: np.ndarray

def main():
    # Initialize parameters
    num_steps = 200
    parameters = Parameters(
        surveillance_region=np.array([[-3000, 3000], [-3000, 3000]]),
        prior_velocity_covariance=np.diag([10**2, 10**2]),
        driving_noise_variance=0.010,
        length_steps=np.ones(num_steps),
        survival_probability=0.999,
        num_sensors=2,
        measurement_variance_range=25**2,
        measurement_variance_bearing=0.5**2,
        detection_probability=0.9,
        measurement_range=12000,  # 2 * surveillance_region max
        mean_clutter=5,
        clutter_distribution=1/(360*12000),
        detection_threshold=0.5,
        threshold_pruning=1e-4,
        minimum_track_length=1,
        num_particles=10000,
        target_start_states=get_start_states(5, 1000, 10),
        target_appearance_from_to=np.array([[5,10,15,20,25], [155,160,165,170,175]]),
        sensor_positions=get_sensor_positions(2, 1000)
    )
    
    # Generate true tracks
    true_tracks = generate_true_tracks(parameters, num_steps)
    
    # Initialize visualizer
    viz = Visualizer(parameters)
    
    # Simulation loop
    for step in range(num_steps):
        print(f"Step {step+1}/{num_steps}")
        
        # Generate measurements
        true_measurements = generate_measurements(true_tracks[:, :, step], parameters)
        cluttered_measurements = generate_cluttered_measurements(true_measurements, parameters)
        
        # Update visualization
        viz.update(true_tracks, cluttered_measurements, step)
    
    plt.show()

    # Apply BP Filter to the measurements
    for step in range(num_steps):
        true_measurements = cluttered_measurements[step]
        if step == 0:
            bp_filter = BPFilter(true_measurements)
        else:
            bp_filter.predict()
            bp_filter.update(true_measurements)
        bp_filter.prune()
        bp_filter.resample()
        bp_filter.estimate()

if __name__ == "__main__":
    np.random.seed(1)
    main()