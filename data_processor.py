import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from threading import Thread, Event
from queue import Queue
import json
from industrial_protocols import IndustrialDataCollector
from ml_module import FactorySyncML
from models import db, Machine, Sensor, SensorReading, DataPoint
import time

class DataProcessor:
    def __init__(self, app):
        self.app = app
        self.data_collector = IndustrialDataCollector(app)
        self.ml_model = FactorySyncML()
        self.processing_queue = Queue()
        self.stop_event = Event()
        self.processing_thread = None
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the data processor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataProcessor')

    def start(self):
        """Start the data processing pipeline."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = Thread(target=self._process_data_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.data_collector.start_collection()
            self.logger.info("Data processing pipeline started")

    def stop(self):
        """Stop the data processing pipeline."""
        self.stop_event.set()
        self.data_collector.stop_collection()
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("Data processing pipeline stopped")

    def _process_data_loop(self):
        """Main data processing loop."""
        while not self.stop_event.is_set():
            try:
                with self.app.app_context():
                    # Get latest readings
                    readings = SensorReading.query.filter(
                        SensorReading.processed == False
                    ).limit(1000).all()

                    if readings:
                        # Process readings in batches
                        self._process_readings_batch(readings)
                        
                        # Mark as processed
                        for reading in readings:
                            reading.processed = True
                        db.session.commit()

                    # Generate insights every hour
                    self._generate_periodic_insights()
                    
                    # Wait before next processing cycle
                    time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error in data processing loop: {str(e)}")
                time.sleep(5)  # Wait before retrying

    def _process_readings_batch(self, readings):
        """Process a batch of sensor readings."""
        try:
            # Convert readings to DataFrame
            data = self._readings_to_dataframe(readings)
            
            # Preprocess data
            processed_data = self.ml_model.preprocess_data(data)
            
            # Detect anomalies
            anomaly_scores = self.ml_model.detect_anomalies(processed_data)
            
            # Predict failures
            failure_probs = self.ml_model.predict_failure_probability(processed_data)
            
            # Update readings with results
            for i, reading in enumerate(readings):
                reading.anomaly_score = float(anomaly_scores[i])
                reading.failure_probability = float(failure_probs[i])
            
            db.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error processing readings batch: {str(e)}")
            db.session.rollback()

    def _readings_to_dataframe(self, readings):
        """Convert sensor readings to a pandas DataFrame."""
        data = []
        for reading in readings:
            row = {
                'machine_id': reading.sensor.machine_id,
                'sensor_id': reading.sensor_id,
                'value': reading.value,
                'timestamp': reading.timestamp
            }
            
            # Add sensor metadata
            row['sensor_name'] = reading.sensor.name
            row['sensor_type'] = reading.sensor.type
            
            # Add machine metadata
            row['machine_type'] = reading.sensor.machine.type
            row['machine_status'] = reading.sensor.machine.status
            
            data.append(row)
        
        return pd.DataFrame(data)

    def _generate_periodic_insights(self):
        """Generate periodic insights from processed data."""
        try:
            current_time = datetime.utcnow()
            last_hour = current_time - timedelta(hours=1)
            
            with self.app.app_context():
                # Get machines
                machines = Machine.query.all()
                
                for machine in machines:
                    # Get recent readings for this machine
                    readings = SensorReading.query.join(Sensor).filter(
                        Sensor.machine_id == machine.id,
                        SensorReading.timestamp > last_hour
                    ).all()
                    
                    if readings:
                        # Convert to DataFrame
                        data = self._readings_to_dataframe(readings)
                        
                        # Calculate KPIs
                        kpis = self.ml_model.calculate_kpis(data)
                        
                        # Generate maintenance schedule if needed
                        failure_probs = [r.failure_probability for r in readings]
                        if max(failure_probs) > 0.5:  # If any high risk
                            schedule = self.ml_model.generate_maintenance_schedule(
                                failure_probs, data
                            )
                            
                            # Store maintenance recommendations
                            self._store_maintenance_recommendations(machine.id, schedule)
                        
                        # Store KPIs
                        self._store_kpis(machine.id, kpis)
                        
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")

    def _store_maintenance_recommendations(self, machine_id, schedule):
        """Store maintenance recommendations in the database."""
        try:
            for rec in schedule:
                data_point = DataPoint(
                    machine_id=machine_id,
                    type='maintenance_recommendation',
                    value=json.dumps(rec),
                    timestamp=datetime.utcnow()
                )
                db.session.add(data_point)
            db.session.commit()
        except Exception as e:
            self.logger.error(f"Error storing maintenance recommendations: {str(e)}")
            db.session.rollback()

    def _store_kpis(self, machine_id, kpis):
        """Store KPIs in the database."""
        try:
            for kpi_name, value in kpis.items():
                data_point = DataPoint(
                    machine_id=machine_id,
                    type=f'kpi_{kpi_name}',
                    value=str(value),
                    timestamp=datetime.utcnow()
                )
                db.session.add(data_point)
            db.session.commit()
        except Exception as e:
            self.logger.error(f"Error storing KPIs: {str(e)}")
            db.session.rollback()

    def get_machine_health_status(self, machine_id):
        """Get current health status for a machine."""
        try:
            # Get recent readings
            recent_readings = SensorReading.query.join(Sensor).filter(
                Sensor.machine_id == machine_id,
                SensorReading.timestamp > datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            if not recent_readings:
                return {
                    'status': 'unknown',
                    'health_score': 0,
                    'alerts': ['No recent data available']
                }
            
            # Calculate health score
            failure_probs = [r.failure_probability for r in recent_readings]
            anomaly_scores = [r.anomaly_score for r in recent_readings]
            
            health_score = 1 - (max(failure_probs) * 0.7 + max(anomaly_scores) * 0.3)
            
            # Generate alerts
            alerts = []
            if max(failure_probs) > 0.8:
                alerts.append('High failure risk detected')
            if max(anomaly_scores) > 0.8:
                alerts.append('Abnormal behavior detected')
            
            # Determine status
            if health_score > 0.8:
                status = 'healthy'
            elif health_score > 0.5:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'health_score': health_score,
                'alerts': alerts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting machine health status: {str(e)}")
            return {
                'status': 'error',
                'health_score': 0,
                'alerts': [f'Error: {str(e)}']
            }
