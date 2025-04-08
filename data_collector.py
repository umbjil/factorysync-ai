import random
import time
from datetime import datetime
import numpy as np
from models import Machine, Sensor, DataPoint, SensorReading, db

class DataCollector:
    def __init__(self, machine_id):
        self.machine = Machine.query.get(machine_id)
        if not self.machine:
            raise ValueError(f"Machine {machine_id} not found")
        
        self.sensors = {s.name: s for s in self.machine.sensors}
        self.current_state = {name: 0 for name in self.sensors}
        self.last_update = datetime.utcnow()
    
    def simulate_sensor_reading(self, sensor):
        """Simulate a sensor reading with some random noise"""
        base_value = (sensor.max_value + sensor.min_value) / 2
        noise = (sensor.max_value - sensor.min_value) * 0.1 * random.gauss(0, 1)
        return max(sensor.min_value, min(sensor.max_value, base_value + noise))
    
    def simulate_anomaly(self, sensor):
        """Simulate an anomalous reading"""
        if random.random() < 0.5:
            return sensor.max_value * (1 + random.random() * 0.2)
        return sensor.min_value * (1 - random.random() * 0.2)
    
    def update_sensor_data(self, data=None):
        """Update sensor data with provided values or simulate new ones"""
        now = datetime.utcnow()
        
        # If no data provided, simulate readings
        if not data:
            data = {}
            for name, sensor in self.sensors.items():
                # Simulate anomaly with small probability
                if random.random() < 0.05:
                    data[name] = self.simulate_anomaly(sensor)
                else:
                    data[name] = self.simulate_sensor_reading(sensor)
        
        # Validate and update current state
        for name, value in data.items():
            if name not in self.sensors:
                raise ValueError(f"Unknown sensor: {name}")
            
            sensor = self.sensors[name]
            # Allow some margin for anomaly detection
            max_allowed = sensor.max_value * 1.5
            min_allowed = sensor.min_value * 0.5
            
            if not (min_allowed <= float(value) <= max_allowed):
                raise ValueError(f"Value {value} for sensor {name} is outside allowed range")
            
            self.current_state[name] = float(value)
        
        self.last_update = now
        return self.current_state.copy()
    
    def get_state(self):
        """Get current state of all sensors"""
        return self.current_state.copy()
    
    def save_data_point(self, is_failure=False, is_anomaly=False, prediction_confidence=None):
        """Save current state as a data point"""
        data_point = DataPoint(
            machine_id=self.machine.id,
            timestamp=self.last_update,
            is_failure=is_failure,
            is_anomaly=is_anomaly,
            prediction_confidence=prediction_confidence
        )
        
        for name, value in self.current_state.items():
            reading = SensorReading(
                sensor_id=self.sensors[name].id,
                value=value
            )
            data_point.readings.append(reading)
        
        db.session.add(data_point)
        db.session.commit()
        return data_point
