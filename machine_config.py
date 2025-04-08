from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os

@dataclass
class SensorConfig:
    name: str
    unit: str
    min_value: float
    max_value: float
    warning_min: float
    warning_max: float

class MachineConfig:
    def __init__(self, machine_id: str, name: str, type: str):
        self.machine_id = machine_id
        self.name = name
        self.type = type
        self.sensors: Dict[str, SensorConfig] = {}
        self.config_dir = "machine_configs"
        
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
    
    def add_sensor(self, name: str, unit: str, min_value: float, max_value: float,
                  warning_min: float = None, warning_max: float = None):
        """Add a new sensor configuration"""
        if warning_min is None:
            warning_min = min_value
        if warning_max is None:
            warning_max = max_value
            
        self.sensors[name] = SensorConfig(
            name=name,
            unit=unit,
            min_value=min_value,
            max_value=max_value,
            warning_min=warning_min,
            warning_max=warning_max
        )
    
    def save(self):
        """Save machine configuration to file"""
        config = {
            "machine_id": self.machine_id,
            "name": self.name,
            "type": self.type,
            "sensors": {
                name: {
                    "unit": sensor.unit,
                    "min_value": sensor.min_value,
                    "max_value": sensor.max_value,
                    "warning_min": sensor.warning_min,
                    "warning_max": sensor.warning_max
                }
                for name, sensor in self.sensors.items()
            }
        }
        
        filename = os.path.join(self.config_dir, f"{self.machine_id}.json")
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load(cls, machine_id: str) -> Optional['MachineConfig']:
        """Load machine configuration from file"""
        config_dir = "machine_configs"
        filename = os.path.join(config_dir, f"{machine_id}.json")
        
        if not os.path.exists(filename):
            return None
            
        with open(filename, "r") as f:
            config = json.load(f)
            
        machine = cls(
            machine_id=config["machine_id"],
            name=config["name"],
            type=config["type"]
        )
        
        for name, sensor_config in config["sensors"].items():
            machine.add_sensor(
                name=name,
                unit=sensor_config["unit"],
                min_value=sensor_config["min_value"],
                max_value=sensor_config["max_value"],
                warning_min=sensor_config["warning_min"],
                warning_max=sensor_config["warning_max"]
            )
        
        return machine
    
    @staticmethod
    def list_machines() -> List[Dict]:
        """List all configured machines"""
        config_dir = "machine_configs"
        if not os.path.exists(config_dir):
            return []
            
        machines = []
        for filename in os.listdir(config_dir):
            if filename.endswith(".json"):
                with open(os.path.join(config_dir, filename), "r") as f:
                    config = json.load(f)
                    machines.append({
                        "machine_id": config["machine_id"],
                        "name": config["name"],
                        "type": config["type"]
                    })
        return machines
