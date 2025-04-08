from machine_config import MachineConfig
from data_collector import MachineDataCollector
from ml_module import ContinuousMLTrainer
import time
import random

def add_custom_machine():
    # Create a new machine configuration
    machine = MachineConfig(
        machine_id="LASER_001",
        name="Laser Cutter",
        type="LASER"
    )
    
    # Add sensors with their specifications
    machine.add_sensor(
        name="laser_power",
        unit="watts",
        min_value=0,
        max_value=500,
        warning_min=50,
        warning_max=450
    )
    
    machine.add_sensor(
        name="focus_distance",
        unit="mm",
        min_value=0,
        max_value=50,
        warning_min=5,
        warning_max=45
    )
    
    machine.add_sensor(
        name="cutting_speed",
        unit="mm/s",
        min_value=0,
        max_value=100,
        warning_min=10,
        warning_max=90
    )
    
    machine.add_sensor(
        name="material_thickness",
        unit="mm",
        min_value=0,
        max_value=20,
        warning_min=0.1,
        warning_max=15
    )
    
    machine.add_sensor(
        name="temperature",
        unit="celsius",
        min_value=0,
        max_value=100,
        warning_min=20,
        warning_max=80
    )
    
    # Save the configuration
    machine.save()
    
    # Initialize data collector and ML trainer
    collector = MachineDataCollector(machine_id="LASER_001")
    trainer = ContinuousMLTrainer(machine_id="LASER_001")
    
    # Simulate some normal operation data
    print("Collecting normal operation data...")
    for i in range(100):
        # Generate normal sensor data
        sensor_data = {
            "laser_power": random.uniform(100, 400),
            "focus_distance": random.uniform(10, 40),
            "cutting_speed": random.uniform(20, 80),
            "material_thickness": random.uniform(1, 10),
            "temperature": random.uniform(30, 70)
        }
        
        # Update collector with new data
        state = collector.update_sensor_data(sensor_data)
        
        # Train the model with this data point
        trainer.add_training_point(state, is_failure=False)
        
        if i % 10 == 0:
            print(f"Processed {i} normal samples")
        time.sleep(0.1)
    
    # Simulate some failure data
    print("\nCollecting failure data...")
    for i in range(20):
        # Generate anomalous sensor data (high temperature)
        sensor_data = {
            "laser_power": random.uniform(100, 400),
            "focus_distance": random.uniform(10, 40),
            "cutting_speed": random.uniform(20, 80),
            "material_thickness": random.uniform(1, 10),
            "temperature": random.uniform(75, 95)  # High temperature
        }
        
        # Update collector with new data
        state = collector.update_sensor_data(sensor_data)
        
        # Train the model with this data point
        trainer.add_training_point(state, is_failure=True)
        
        if i % 5 == 0:
            print(f"Processed {i} failure samples")
        time.sleep(0.1)
    
    # Get final metrics
    metrics = trainer.get_metrics()
    print("\nFinal model metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test prediction
    test_data = {
        "laser_power": 380,
        "focus_distance": 35,
        "cutting_speed": 75,
        "material_thickness": 8,
        "temperature": 85  # High temperature
    }
    
    prediction = trainer.predict(test_data)
    print("\nTest prediction for high temperature:")
    print(f"Failure predicted: {prediction['failure_predicted']}")
    print(f"Anomaly detected: {prediction['anomaly_detected']}")
    print(f"Confidence: {prediction['confidence']:.3f}")

if __name__ == "__main__":
    add_custom_machine()
