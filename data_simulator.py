import time
import random
import math
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Machine, Sensor, SensorReading, db
import numpy as np
from threading import Thread, Event

class DataSimulator:
    def __init__(self, app):
        self.app = app
        self.stop_event = Event()
        self.simulation_thread = None
        
    def start(self):
        """Start the simulation in a separate thread."""
        if self.simulation_thread is None or not self.simulation_thread.is_alive():
            self.stop_event.clear()
            self.simulation_thread = Thread(target=self._simulate_data)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            print("Data simulation started")
    
    def stop(self):
        """Stop the simulation."""
        self.stop_event.set()
        if self.simulation_thread:
            self.simulation_thread.join()
            print("Data simulation stopped")

    def _generate_sensor_value(self, sensor, current_value=None):
        """Generate a realistic sensor value based on sensor type and limits."""
        min_val = sensor.min_value
        max_val = sensor.max_value
        
        # If no current value, start at a reasonable point
        if current_value is None:
            current_value = (min_val + max_val) / 2
        
        # Different patterns for different sensor types
        if sensor.type == 'Temperature':
            # Temperature changes slowly with some random fluctuation
            change = random.uniform(-2, 2)
            new_value = current_value + change
        
        elif sensor.type == 'RPM':
            # RPM can change more quickly but tends to maintain a range
            change = random.uniform(-100, 100)
            new_value = current_value + change
        
        elif sensor.type == 'Pressure':
            # Pressure changes moderately with occasional spikes
            if random.random() < 0.05:  # 5% chance of pressure spike
                new_value = current_value + random.uniform(-50, 50)
            else:
                new_value = current_value + random.uniform(-10, 10)
        
        elif sensor.type == 'Force':
            # Force varies with regular patterns plus noise
            base = (math.sin(time.time() * 0.1) + 1) / 2  # Oscillating base value
            noise = random.uniform(-0.1, 0.1)
            new_value = min_val + (max_val - min_val) * (base + noise)
        
        elif sensor.type == 'Position' or sensor.type == 'Angle':
            # Position/Angle often moves between limits
            if random.random() < 0.1:  # 10% chance to start moving to a new position
                new_value = random.uniform(min_val, max_val)
            else:
                new_value = current_value + random.uniform(-5, 5)
        
        elif sensor.type == 'Power':
            # Power consumption often follows a pattern
            hour = datetime.now().hour
            # Higher power usage during working hours (8-18)
            base_load = 0.7 if 8 <= hour <= 18 else 0.3
            variation = random.uniform(-0.1, 0.1)
            new_value = min_val + (max_val - min_val) * (base_load + variation)
        
        elif sensor.type == 'Speed':
            # Speed changes gradually with occasional adjustments
            if random.random() < 0.1:  # 10% chance of speed adjustment
                new_value = current_value + random.uniform(-20, 20)
            else:
                new_value = current_value + random.uniform(-5, 5)
        
        elif sensor.type == 'Flow':
            # Flow rate varies smoothly
            change = random.uniform(-2, 2)
            new_value = current_value + change
        
        elif sensor.type == 'Humidity':
            # Humidity changes very slowly
            change = random.uniform(-1, 1)
            new_value = current_value + change
        
        else:
            # Default pattern for other sensor types
            change = random.uniform(-5, 5)
            new_value = current_value + change
        
        # Ensure value stays within limits
        new_value = max(min_val, min(max_val, new_value))
        return new_value

    def _simulate_machine_behavior(self, machine, sensor_values):
        """Simulate realistic machine behavior and update sensor values."""
        # Initialize sensor values if not present
        if machine.id not in sensor_values:
            sensor_values[machine.id] = {}
            for sensor in machine.sensors:
                sensor_values[machine.id][sensor.id] = (sensor.min_value + sensor.max_value) / 2

        # Update each sensor value
        for sensor in machine.sensors:
            current_value = sensor_values[machine.id][sensor.id]
            new_value = self._generate_sensor_value(sensor, current_value)
            sensor_values[machine.id][sensor.id] = new_value
            
            # Create a new sensor reading
            reading = SensorReading(
                sensor_id=sensor.id,
                value=new_value,
                timestamp=datetime.utcnow()
            )
            
            # Calculate anomaly score (simple example - you can make this more sophisticated)
            range_size = sensor.max_value - sensor.min_value
            center = (sensor.max_value + sensor.min_value) / 2
            distance_from_center = abs(new_value - center)
            reading.anomaly_score = distance_from_center / (range_size / 2)
            
            yield reading

    def _simulate_data(self):
        """Main simulation loop."""
        with self.app.app_context():
            sensor_values = {}  # Keep track of current values
            
            while not self.stop_event.is_set():
                try:
                    # Get all machines
                    machines = Machine.query.all()
                    
                    for machine in machines:
                        # Generate and save new readings
                        new_readings = list(self._simulate_machine_behavior(machine, sensor_values))
                        db.session.bulk_save_objects(new_readings)
                        
                    db.session.commit()
                    print(f"Generated data for {len(machines)} machines at {datetime.now()}")
                    
                except Exception as e:
                    print(f"Error in data simulation: {str(e)}")
                    db.session.rollback()
                
                # Wait before next update (5 seconds)
                time.sleep(5)
