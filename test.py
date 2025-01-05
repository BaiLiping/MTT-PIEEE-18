def tracker_bp(cluttered_measurements, legacy_pts, legacy_existences, legacy_labels,
               unknown_number, unknown_particles, step, parameters):
    """
    Python implementation of the TrackerBP algorithm.
    
    Args:
        cluttered_measurements: List of measurements for each sensor
        legacy_pts: Previous particle states
        legacy_existences: Previous existence probabilities
        legacy_labels: Previous target labels
        unknown_number: Unknown parameter
        unknown_particles: Unknown parameter
        step: Current time step
        parameters: Dictionary containing algorithm parameters
    
    Returns:
        tuple: (estimates, estimated_cardinality, legacy_pts, legacy_existences, legacy_labels)
    """
    
    num_sensors = parameters['num_sensors']
    detection_threshold = parameters['detection_threshold']
    threshold_pruning = parameters['threshold_pruning']
    length_step = parameters['length_steps'][step]
    
    # Perform prediction
    legacy_pts, legacy_existences = perform_prediction(legacy_pts, legacy_existences, 
                                                     length_step, parameters)
    
    # Process each sensor
    for sensor in range(num_sensors):
        measurements = cluttered_measurements[sensor]
        
        # Introduce new PTs
        new_pts, new_labels, new_existences, xi_messages = introduce_new_pts(
            measurements, sensor, step, unknown_number, unknown_particles, parameters)
        
        # Evaluate v factors
        beta_messages, v_factors1 = evaluate_measurements(
            legacy_pts, legacy_existences, measurements, parameters, sensor)
        
        # Perform iterative data association
        kappas, iotas = perform_data_association_bp(beta_messages, xi_messages, 20, 1e-5, 1e5)
        
        # Update PTs
        legacy_pts, legacy_existences, legacy_labels = update_pts(
            kappas, iotas, legacy_pts, new_pts, legacy_existences, new_existences,
            legacy_labels, new_labels, v_factors1)
        
        # Perform pruning
        num_targets = legacy_pts.shape[2]
        is_redundant = np.zeros(num_targets, dtype=bool)
        
        for target in range(num_targets):
            if legacy_existences[target] < threshold_pruning:
                is_redundant[target] = True
        
        # Remove redundant targets
        legacy_pts = legacy_pts[:, :, ~is_redundant]
        legacy_labels = legacy_labels[:, ~is_redundant]
        legacy_existences = legacy_existences[~is_redundant]
    
    # Calculate estimated cardinality
    estimated_cardinality = np.sum(legacy_existences)
    
    # Perform estimation
    num_targets = legacy_pts.shape[2]
    detected_targets = 0
    estimates = {'state': [], 'label': []}
    
    for target in range(num_targets):
        if legacy_existences[target] > detection_threshold:
            detected_targets += 1
            target_state = np.mean(legacy_pts[:, :, target], axis=1)
            target_label = legacy_labels[:, target]
            
            if detected_targets == 1:
                estimates['state'] = target_state.reshape(-1, 1)
                estimates['label'] = target_label.reshape(-1, 1)
            else:
                estimates['state'] = np.column_stack((estimates['state'], target_state))
                estimates['label'] = np.column_stack((estimates['label'], target_label))
    
    return estimates, estimated_cardinality, legacy_pts, legacy_existences, legacy_labels