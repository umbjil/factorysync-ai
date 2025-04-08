import time
from ml_module import ContinuousMLTrainer
from data_collector import MachineDataCollector
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_machine_data(collector, duration=30):
    """Analyze and plot machine sensor data"""
    data_points = []
    start_time = datetime.now()
    
    print(f"\nCollecting {duration} seconds of data for analysis...")
    for _ in range(duration):
        state = collector.get_current_state()
        data_points.append(state)
        time.sleep(1)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data_points)
    
    # Calculate correlations
    sensor_cols = ['temperature', 'pressure', 'vibration', 'rpm', 'power_consumption', 
                  'current', 'air_quality', 'voc_level', 'ultrasonic', 'noise_level', 
                  'oil_level', 'humidity']
    
    print("\nSensor Correlations:")
    corr = df[sensor_cols].corr()
    print(corr.round(2))
    
    # Plot sensor trends
    plt.figure(figsize=(15, 10))
    for i, sensor in enumerate(sensor_cols[:6], 1):  # Plot first 6 main sensors
        plt.subplot(2, 3, i)
        plt.plot(df[sensor])
        plt.title(f'{sensor} Trend')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensor_trends.png')
    plt.close()
    
    return df

def generate_training_data():
    """Generate training data with various machine conditions"""
    print("Initializing ML trainer and data collectors...")
    ml_trainer = ContinuousMLTrainer()
    collectors = [
        MachineDataCollector(machine_id=i) for i in range(3)
    ]
    
    try:
        # Start data collection
        print("Starting data collection...")
        for collector in collectors:
            collector.start_collection()
        
        # Analyze normal operation data
        print("\nAnalyzing normal operations...")
        normal_data = analyze_machine_data(collectors[0])
        
        # Run various scenarios
        print("\nSimulating various machine conditions...")
        for scenario in range(5):
            # Randomly select a machine and condition
            machine_id = random.randint(0, len(collectors)-1)
            condition = random.choice(['high_load', 'overheating', 'vibration_fault', 'pressure_drop'])
            
            print(f"\nScenario {scenario + 1}: Simulating {condition} on machine {machine_id}")
            
            # Apply specific condition
            if condition == 'high_load':
                collectors[machine_id].state['power_consumption'] = 58.0
                collectors[machine_id].state['current'] = 11.5
                collectors[machine_id].state['temperature'] = 32.0
            elif condition == 'overheating':
                collectors[machine_id].state['temperature'] = 34.0
                collectors[machine_id].state['oil_level'] = 75.0
            elif condition == 'vibration_fault':
                collectors[machine_id].state['vibration'] = 0.28
                collectors[machine_id].state['noise_level'] = 82.0
            elif condition == 'pressure_drop':
                collectors[machine_id].state['pressure'] = 88.0
                collectors[machine_id].state['rpm'] = 920.0
            
            # Simulate anomaly
            collectors[machine_id].simulate_anomaly(duration=30)
            
            # Analyze anomaly data
            print(f"Analyzing {condition} condition...")
            anomaly_data = analyze_machine_data(collectors[machine_id])
            
            # Monitor all machines
            for _ in range(20):  # Monitor for 20 seconds
                for i, collector in enumerate(collectors):
                    state = collector.get_current_state()
                    prediction = ml_trainer.predict_failure_probability(state)
                    if prediction and prediction['failure_probability'] > 0.5:
                        print(f"Machine {i} - High Failure Risk: {prediction['failure_probability']:.2f}")
                        print("Critical Sensors:")
                        for sensor, value in state.items():
                            if isinstance(value, (int, float)) and sensor != 'timestamp':
                                print(f"  {sensor}: {value:.2f}")
                    ml_trainer.add_training_data(state)
                time.sleep(1)
            
            # Perform maintenance if needed
            state = collectors[machine_id].get_current_state()
            if state['failure_probability'] > 0.7:
                print(f"\nPerforming maintenance on machine {machine_id}")
                collectors[machine_id].perform_maintenance()
            
            # Let system stabilize
            print("Letting system stabilize...")
            time.sleep(20)
        
        # Get final metrics
        print("\nFinal ML Model Metrics:")
        metrics = ml_trainer.get_model_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Print data summary
        print("\nData Collection Summary:")
        for i, collector in enumerate(collectors):
            state = collector.get_current_state()
            print(f"\nMachine {i}:")
            print(f"Total Operating Hours: {state['total_operating_hours']:.2f}")
            print(f"Hours Since Maintenance: {state['hours_since_maintenance']:.2f}")
            print(f"Current Failure Probability: {state['failure_probability']:.2f}")
            print("\nSensor Status:")
            for sensor, value in state.items():
                if isinstance(value, (int, float)) and sensor not in ['timestamp', 'last_maintenance']:
                    print(f"  {sensor}: {value:.2f}")
    
    finally:
        # Cleanup
        print("\nStopping data collection...")
        for collector in collectors:
            collector.stop_collection()
        ml_trainer.stop()
        print("Test completed.")

if __name__ == "__main__":
    generate_training_data()
