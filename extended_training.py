import logging
import time
from data_collector import MachineDataCollector
from ml_module import ContinuousMLTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_machine(machine_type, machine_name):
    """Train models for a specific machine type"""
    logging.info(f"\n{'='*50}")
    logging.info(f"Starting training for {machine_name}")
    logging.info(f"{'='*50}\n")

    # Initialize collector and trainer
    collector = MachineDataCollector(machine_type=machine_type)
    trainer = ContinuousMLTrainer(machine_type=machine_type)

    # Collect initial training data
    num_samples = 1000  # Increased sample size
    normal_samples = int(num_samples * 0.7)  # 70% normal operation
    anomaly_samples = num_samples - normal_samples  # 30% anomalies

    # Collect normal operation data
    logging.info(f"Collecting {normal_samples} normal operation samples...")
    for i in range(normal_samples):
        collector.simulate_operation(duration=1)
        state = collector.get_state()
        trainer.add_training_point(state, state.get('machine_failure', False))
        if i % 100 == 0:  # Progress update
            logging.info(f"Collected {i}/{normal_samples} normal samples")
        time.sleep(0.1)

    # Collect anomaly data with forced failures
    logging.info(f"Collecting {anomaly_samples} anomaly samples...")
    for i in range(anomaly_samples):
        collector.simulate_anomaly(duration=1)
        state = collector.get_state()
        state['machine_failure'] = True  # Force failure state
        trainer.add_training_point(state, True)
        if i % 100 == 0:  # Progress update
            logging.info(f"Collected {i}/{anomaly_samples} anomaly samples")
        time.sleep(0.1)

    # Get final metrics
    metrics = trainer.get_metrics()
    logging.info(f"\nFinal metrics for {machine_type}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logging.info(f"{key}: {value:.3f}")
        else:
            logging.info(f"{key}: {value}")

    logging.info(f"\n{'='*50}")
    logging.info(f"Completed training for {machine_name}")
    logging.info(f"Final metrics: {metrics}")
    logging.info(f"{'='*50}\n")

    return metrics

def main():
    """Main training function"""
    machine_configs = [
        ('L', 'CNC Machine'),
        ('M', 'Drilling Machine'),
        ('H', 'Grinding Machine'),
        ('C', 'Conveyor Belt'),
        ('R', 'Robotic System')
    ]

    all_metrics = {}
    for machine_type, machine_name in machine_configs:
        metrics = train_machine(machine_type, machine_name)
        all_metrics[machine_name] = metrics

    # Print overall summary
    logging.info(f"\nTraining Summary:")
    logging.info(f"{'='*50}")
    for machine_name, metrics in all_metrics.items():
        logging.info(f"\n{machine_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.3f}")
            else:
                logging.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
