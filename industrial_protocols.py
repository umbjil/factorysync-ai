import asyncio
from asyncua import Client as OPCUAClient
from pymodbus.client import ModbusTcpClient
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import threading
from models import db, Machine, Sensor, SensorReading

class MTConnectAdapter:
    def __init__(self, base_url):
        self.base_url = base_url
        self.current = f"{base_url}/current"
        self.probe = f"{base_url}/probe"

    def get_devices(self):
        """Get available devices and their data items."""
        try:
            response = requests.get(self.probe)
            root = ET.fromstring(response.content)
            devices = []
            
            for device in root.findall(".//Device"):
                device_data = {
                    'name': device.get('name'),
                    'uuid': device.get('uuid'),
                    'data_items': []
                }
                
                for item in device.findall(".//DataItem"):
                    device_data['data_items'].append({
                        'id': item.get('id'),
                        'type': item.get('type'),
                        'category': item.get('category')
                    })
                    
                devices.append(device_data)
            
            return devices
        except Exception as e:
            print(f"Error getting MTConnect devices: {str(e)}")
            return []

    def get_current_data(self):
        """Get current values for all data items."""
        try:
            response = requests.get(self.current)
            root = ET.fromstring(response.content)
            data = {}
            
            for item in root.findall(".//DataItem"):
                data[item.get('dataItemId')] = {
                    'value': item.text,
                    'timestamp': item.get('timestamp')
                }
            
            return data
        except Exception as e:
            print(f"Error getting MTConnect data: {str(e)}")
            return {}

class OPCUAAdapter:
    def __init__(self, url):
        self.url = url
        self.client = None
        self._running = False
        self._thread = None

    async def connect(self):
        """Connect to OPC UA server."""
        try:
            self.client = OPCUAClient(self.url)
            await self.client.connect()
            print(f"Connected to OPC UA server at {self.url}")
            return True
        except Exception as e:
            print(f"Error connecting to OPC UA server: {str(e)}")
            return False

    async def browse_nodes(self):
        """Browse available nodes on the server."""
        try:
            root = self.client.get_root_node()
            nodes = await root.get_children()
            return [{'node_id': node.nodeid.to_string(), 
                    'name': (await node.read_browse_name()).Name} 
                   for node in nodes]
        except Exception as e:
            print(f"Error browsing OPC UA nodes: {str(e)}")
            return []

    async def read_node(self, node_id):
        """Read value from a specific node."""
        try:
            node = self.client.get_node(node_id)
            value = await node.read_value()
            return value
        except Exception as e:
            print(f"Error reading OPC UA node: {str(e)}")
            return None

class ModbusAdapter:
    def __init__(self, host, port=502):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        """Connect to Modbus TCP server."""
        try:
            self.client = ModbusTcpClient(self.host, self.port)
            return self.client.connect()
        except Exception as e:
            print(f"Error connecting to Modbus server: {str(e)}")
            return False

    def read_holding_registers(self, address, count=1, unit=1):
        """Read holding registers."""
        try:
            result = self.client.read_holding_registers(address, count, unit=unit)
            if not result.isError():
                return result.registers
            return None
        except Exception as e:
            print(f"Error reading Modbus registers: {str(e)}")
            return None

    def read_input_registers(self, address, count=1, unit=1):
        """Read input registers."""
        try:
            result = self.client.read_input_registers(address, count, unit=unit)
            if not result.isError():
                return result.registers
            return None
        except Exception as e:
            print(f"Error reading Modbus registers: {str(e)}")
            return None

class IndustrialDataCollector:
    def __init__(self, app):
        self.app = app
        self.mtconnect_adapters = {}
        self.opcua_adapters = {}
        self.modbus_adapters = {}
        self.stop_event = threading.Event()
        self.collection_thread = None

    def add_mtconnect_source(self, name, url):
        """Add an MTConnect data source."""
        self.mtconnect_adapters[name] = MTConnectAdapter(url)

    def add_opcua_source(self, name, url):
        """Add an OPC UA data source."""
        self.opcua_adapters[name] = OPCUAAdapter(url)

    def add_modbus_source(self, name, host, port=502):
        """Add a Modbus TCP data source."""
        self.modbus_adapters[name] = ModbusAdapter(host, port)

    async def collect_opcua_data(self):
        """Collect data from all OPC UA sources."""
        for name, adapter in self.opcua_adapters.items():
            if await adapter.connect():
                nodes = await adapter.browse_nodes()
                for node in nodes:
                    value = await adapter.read_node(node['node_id'])
                    if value is not None:
                        # Store in database
                        with self.app.app_context():
                            try:
                                # Find or create machine
                                machine = Machine.query.filter_by(name=name).first()
                                if not machine:
                                    machine = Machine(name=name, type='OPC UA')
                                    db.session.add(machine)
                                    db.session.flush()

                                # Find or create sensor
                                sensor = Sensor.query.filter_by(
                                    machine_id=machine.id,
                                    name=node['name']
                                ).first()
                                if not sensor:
                                    sensor = Sensor(
                                        machine_id=machine.id,
                                        name=node['name'],
                                        type='OPC UA'
                                    )
                                    db.session.add(sensor)
                                    db.session.flush()

                                # Create reading
                                reading = SensorReading(
                                    sensor_id=sensor.id,
                                    value=float(value),
                                    timestamp=datetime.utcnow()
                                )
                                db.session.add(reading)
                                db.session.commit()
                            except Exception as e:
                                print(f"Error storing OPC UA data: {str(e)}")
                                db.session.rollback()

    def collect_modbus_data(self):
        """Collect data from all Modbus sources."""
        for name, adapter in self.modbus_adapters.items():
            if adapter.connect():
                # Example: read first 10 holding registers
                values = adapter.read_holding_registers(0, 10)
                if values:
                    with self.app.app_context():
                        try:
                            # Find or create machine
                            machine = Machine.query.filter_by(name=name).first()
                            if not machine:
                                machine = Machine(name=name, type='Modbus')
                                db.session.add(machine)
                                db.session.flush()

                            # Create sensors and readings for each register
                            for i, value in enumerate(values):
                                sensor = Sensor.query.filter_by(
                                    machine_id=machine.id,
                                    name=f"Register_{i}"
                                ).first()
                                if not sensor:
                                    sensor = Sensor(
                                        machine_id=machine.id,
                                        name=f"Register_{i}",
                                        type='Modbus'
                                    )
                                    db.session.add(sensor)
                                    db.session.flush()

                                reading = SensorReading(
                                    sensor_id=sensor.id,
                                    value=float(value),
                                    timestamp=datetime.utcnow()
                                )
                                db.session.add(reading)
                            db.session.commit()
                        except Exception as e:
                            print(f"Error storing Modbus data: {str(e)}")
                            db.session.rollback()

    def collect_mtconnect_data(self):
        """Collect data from all MTConnect sources."""
        for name, adapter in self.mtconnect_adapters.items():
            data = adapter.get_current_data()
            if data:
                with self.app.app_context():
                    try:
                        # Find or create machine
                        machine = Machine.query.filter_by(name=name).first()
                        if not machine:
                            machine = Machine(name=name, type='MTConnect')
                            db.session.add(machine)
                            db.session.flush()

                        # Create sensors and readings for each data item
                        for item_id, item_data in data.items():
                            sensor = Sensor.query.filter_by(
                                machine_id=machine.id,
                                name=item_id
                            ).first()
                            if not sensor:
                                sensor = Sensor(
                                    machine_id=machine.id,
                                    name=item_id,
                                    type='MTConnect'
                                )
                                db.session.add(sensor)
                                db.session.flush()

                            try:
                                value = float(item_data['value'])
                                reading = SensorReading(
                                    sensor_id=sensor.id,
                                    value=value,
                                    timestamp=datetime.utcnow()
                                )
                                db.session.add(reading)
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass

                        db.session.commit()
                    except Exception as e:
                        print(f"Error storing MTConnect data: {str(e)}")
                        db.session.rollback()

    def start_collection(self):
        """Start collecting data from all sources."""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.stop_event.clear()
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            print("Industrial data collection started")

    def stop_collection(self):
        """Stop collecting data."""
        self.stop_event.set()
        if self.collection_thread:
            self.collection_thread.join()
            print("Industrial data collection stopped")

    def _collection_loop(self):
        """Main collection loop."""
        while not self.stop_event.is_set():
            # Collect MTConnect data
            self.collect_mtconnect_data()

            # Collect Modbus data
            self.collect_modbus_data()

            # Collect OPC UA data
            asyncio.run(self.collect_opcua_data())

            # Wait before next collection
            time.sleep(5)  # Adjust collection interval as needed
