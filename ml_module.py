from datetime import datetime
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from machine_config import MachineConfig
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from threading import Lock

class ContinuousMLTrainer:
    def __init__(self, machine_id: str):
        """Initialize ML trainer for a specific machine"""
        self.machine_id = machine_id
        self.machine_config = MachineConfig.load(machine_id)
        if not self.machine_config:
            raise ValueError(f"No configuration found for machine {machine_id}")
            
        self.training_data = []
        self.feature_names = None
        self.scaler = None
        self.models = {
            'failure_classifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'anomaly_detector': IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
        }
        
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'anomaly_rate': 0.0,
            'last_training': None,
            'samples_processed': 0
        }
        
        # Create model directory if it doesn't exist
        self.model_dir = os.path.join('models', machine_id)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Try to load existing models
        try:
            self._load_models()
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
    
    def _engineer_features(self, data_point):
        """Engineer features from raw sensor data"""
        features = {}
        
        # Add raw sensor values
        for sensor_name in self.machine_config.sensors:
            if sensor_name in data_point:
                features[f"{sensor_name}_raw"] = data_point[sensor_name]
                
                # Add normalized values (relative to warning thresholds)
                sensor_config = self.machine_config.sensors[sensor_name]
                warning_range = sensor_config.warning_max - sensor_config.warning_min
                if warning_range > 0:
                    normalized_value = (data_point[sensor_name] - sensor_config.warning_min) / warning_range
                    features[f"{sensor_name}_normalized"] = normalized_value
        
        # Add timestamp-based features if available
        if 'timestamp' in data_point:
            timestamp = datetime.fromisoformat(data_point['timestamp'])
            features['hour_of_day'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
        
        return features
    
    def add_training_point(self, data_point, is_failure):
        """Add a new training point"""
        self.training_data.append((data_point.copy(), bool(is_failure)))
        
        # Retrain if we have enough data
        if len(self.training_data) >= 10:
            self._train_models()
    
    def predict(self, data_point):
        """Make predictions for a single data point"""
        if not self.feature_names or not self.scaler:
            return {
                'failure_predicted': False,
                'anomaly_detected': False,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
        # Extract features
        features = self._engineer_features(data_point)
        
        # Create feature vector with consistent ordering
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([feature_vector])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        failure_pred = self.models['failure_classifier'].predict(X_scaled)[0]
        failure_prob = self.models['failure_classifier'].predict_proba(X_scaled)[0]
        anomaly_pred = self.models['anomaly_detector'].predict(X_scaled)[0]
        
        return {
            'failure_predicted': bool(failure_pred),
            'anomaly_detected': anomaly_pred == -1,
            'confidence': float(max(failure_prob)),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self):
        """Get current model metrics"""
        return self.metrics.copy()
    
    def _save_models(self):
        """Save trained models and scaler"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        joblib.dump(self.models, os.path.join(self.model_dir, 'models.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'feature_names.joblib'))
        logging.info("Models saved successfully")
    
    def _load_models(self):
        """Load trained models and scaler"""
        models_path = os.path.join(self.model_dir, 'models.joblib')
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        feature_names_path = os.path.join(self.model_dir, 'feature_names.joblib')
        
        if os.path.exists(models_path):
            self.models = joblib.load(models_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feature_names_path)
    
    def _train_models(self):
        """Train machine learning models on collected data"""
        if len(self.training_data) < 10:
            return  # Need at least 10 samples to train

        # Convert training data to numpy arrays with engineered features
        X = []
        y = []
        feature_names = None

        # First pass: collect all possible feature names
        for data_point, _ in self.training_data:
            features = self._engineer_features(data_point)
            if feature_names is None:
                feature_names = sorted(features.keys())
            else:
                feature_names = sorted(set(feature_names) | set(features.keys()))

        # Second pass: create feature vectors with consistent ordering
        for data_point, is_failure in self.training_data:
            features = self._engineer_features(data_point)
            feature_vector = [features.get(name, 0.0) for name in feature_names]  # Use 0.0 for missing features
            X.append(feature_vector)
            y.append(1 if is_failure else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have both positive and negative samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return  # Need both normal and failure cases
            
        # Check class balance
        class_counts = np.bincount(y)
        if min(class_counts) < 2:
            return  # Need at least 2 samples per class
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Initialize base classifier with balanced class weights
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Train classifier
        base_classifier.fit(X_train_scaled, y_train)
        self.models['failure_classifier'] = base_classifier
        
        # Train anomaly detector on normal samples only
        normal_samples = X_train_scaled[y_train == 0]
        self.models['anomaly_detector'].fit(normal_samples)
        
        # Update metrics using validation set
        y_pred = self.models['failure_classifier'].predict(X_val_scaled)
        
        # Calculate metrics with handling for edge cases
        precision = np.mean(y_pred[y_val == 1] == 1) if any(y_val == 1) else 0
        recall = np.mean(y_pred[y_val == 0] == 0) if any(y_val == 0) else 0
        
        self.metrics.update({
            'accuracy': np.mean(y_pred == y_val),
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'anomaly_rate': np.mean(self.models['anomaly_detector'].predict(X_val_scaled) == -1),
            'last_training': datetime.now().isoformat(),
            'samples_processed': len(self.training_data)
        })
        
        # Save feature names for prediction
        self.feature_names = feature_names
        
        # Save models and scaler
        self.scaler = scaler
        self._save_models()
        
        # Log training results
        logging.info(f"Training completed with metrics:")
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                logging.info(f"{metric}: {value:.3f}")
            else:
                logging.info(f"{metric}: {value}")

class FactorySyncML:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_columns = [
            'Type', 'Air_temperature_K', 'Process_temperature_K',
            'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
            'Machine_status', 'Vibration_mm_s', 'Power_kW',
            'Pressure_bar', 'Flow_rate_l_min'
        ]
        self.categorical_features = ['Type', 'Machine_status']
        self.numerical_features = [col for col in self.feature_columns 
                                 if col not in self.categorical_features]
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.model_lock = Lock()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the ML module."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_module.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FactorySyncML')

    def preprocess_data(self, data):
        """
        Preprocess raw machine data for model input.
        
        Args:
            data (pd.DataFrame): Raw data with machine readings
            
        Returns:
            pd.DataFrame: Preprocessed data ready for model
        """
        try:
            df = data.copy()
            
            # Handle missing values
            for col in self.numerical_features:
                if col in df.columns:
                    # Use domain-specific fallbacks
                    if col == 'Process_temperature_K' and 'Air_temperature_K' in df.columns:
                        df[col].fillna(df['Air_temperature_K'] + 10, inplace=True)
                    elif col == 'Vibration_mm_s':
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)

            # Convert units if needed
            if 'Temperature_C' in df.columns:
                df['Temperature_K'] = df['Temperature_C'] + 273.15
            
            # Handle categorical features
            for cat_col in self.categorical_features:
                if cat_col in df.columns:
                    df[cat_col] = self.label_encoder.fit_transform(df[cat_col])
            
            # Scale numerical features
            numerical_data = df[self.numerical_features]
            df[self.numerical_features] = self.scaler.fit_transform(numerical_data)
            
            # Feature engineering
            if 'Tool_wear_min' in df.columns and 'timestamp' in df.columns:
                df['wear_rate'] = df.groupby('machine_id')['Tool_wear_min'].diff() / \
                                df.groupby('machine_id')['timestamp'].diff().dt.total_seconds()
            
            if 'Rotational_speed_rpm' in df.columns and 'Power_kW' in df.columns:
                df['efficiency'] = df['Power_kW'] / (df['Rotational_speed_rpm'] + 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def detect_anomalies(self, data):
        """
        Detect anomalies in machine data using Isolation Forest.
        
        Args:
            data (pd.DataFrame): Preprocessed machine data
            
        Returns:
            np.array: Anomaly scores (-1 for anomalies, 1 for normal)
        """
        try:
            numerical_data = data[self.numerical_features]
            return self.anomaly_detector.fit_predict(numerical_data)
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """
        Train the failure prediction model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels (0: normal, 1: failure)
        """
        try:
            with self.model_lock:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X_train, y_train)
                self.logger.info("Model training completed successfully")
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict_failure_probability(self, data):
        """
        Predict failure probability for machine data.
        
        Args:
            data (pd.DataFrame): Preprocessed machine data
            
        Returns:
            np.array: Failure probabilities
        """
        try:
            with self.model_lock:
                if self.model is None:
                    raise ValueError("Model not trained")
                return self.model.predict_proba(data)[:, 1]
        except Exception as e:
            self.logger.error(f"Error in failure prediction: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Performance metrics
        """
        try:
            with self.model_lock:
                y_pred = self.model.predict(X_test)
                return {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred)
                }
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def save_model(self, path):
        """Save the trained model to disk."""
        try:
            with self.model_lock:
                if self.model is None:
                    raise ValueError("No model to save")
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'feature_columns': self.feature_columns
                }, path)
                self.logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path):
        """Load a trained model from disk."""
        try:
            with self.model_lock:
                saved_data = joblib.load(path)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.label_encoder = saved_data['label_encoder']
                self.feature_columns = saved_data['feature_columns']
                self.logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_maintenance_schedule(self, predictions, machine_data):
        """
        Generate maintenance schedule based on failure predictions.
        
        Args:
            predictions (np.array): Failure probabilities
            machine_data (pd.DataFrame): Machine information
            
        Returns:
            list: Maintenance recommendations
        """
        try:
            schedule = []
            high_risk_threshold = 0.8
            
            for i, prob in enumerate(predictions):
                if prob > high_risk_threshold:
                    machine_id = machine_data.iloc[i]['machine_id']
                    machine_type = machine_data.iloc[i]['Type']
                    
                    # Calculate recommended maintenance window
                    current_time = datetime.now()
                    maintenance_window = {
                        'start': current_time + timedelta(hours=24),  # Start tomorrow
                        'end': current_time + timedelta(hours=48)     # Within 48 hours
                    }
                    
                    schedule.append({
                        'machine_id': machine_id,
                        'machine_type': machine_type,
                        'failure_probability': prob,
                        'recommended_window': maintenance_window,
                        'priority': 'HIGH' if prob > 0.9 else 'MEDIUM',
                        'estimated_downtime': '2 hours',  # This could be made more sophisticated
                        'maintenance_type': self._determine_maintenance_type(machine_data.iloc[i])
                    })
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error generating maintenance schedule: {str(e)}")
            raise

    def _determine_maintenance_type(self, machine_data):
        """Determine the type of maintenance needed based on machine data."""
        maintenance_types = []
        
        # Check various conditions
        if 'Tool_wear_min' in machine_data and machine_data['Tool_wear_min'] > 180:
            maintenance_types.append('Tool Replacement')
        
        if 'Vibration_mm_s' in machine_data and machine_data['Vibration_mm_s'] > 5:
            maintenance_types.append('Vibration Check')
        
        if 'Temperature_K' in machine_data and machine_data['Temperature_K'] > 350:
            maintenance_types.append('Cooling System Check')
        
        return maintenance_types if maintenance_types else ['General Inspection']

    def calculate_kpis(self, machine_data):
        """
        Calculate Key Performance Indicators for machines.
        
        Args:
            machine_data (pd.DataFrame): Historical machine data
            
        Returns:
            dict: KPIs including OEE, availability, performance, and quality
        """
        try:
            # Calculate OEE components
            availability = self._calculate_availability(machine_data)
            performance = self._calculate_performance(machine_data)
            quality = self._calculate_quality(machine_data)
            
            # Calculate OEE
            oee = availability * performance * quality
            
            return {
                'oee': oee,
                'availability': availability,
                'performance': performance,
                'quality': quality,
                'mtbf': self._calculate_mtbf(machine_data),
                'mttr': self._calculate_mttr(machine_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {str(e)}")
            raise

    def _calculate_availability(self, data):
        """Calculate machine availability."""
        try:
            planned_production_time = data['runtime'].sum()
            downtime = data['downtime'].sum()
            return (planned_production_time - downtime) / planned_production_time if planned_production_time > 0 else 0
        except:
            return 0.95  # Default if data not available

    def _calculate_performance(self, data):
        """Calculate machine performance."""
        try:
            actual_output = data['actual_output'].sum()
            theoretical_output = data['theoretical_output'].sum()
            return actual_output / theoretical_output if theoretical_output > 0 else 0
        except:
            return 0.85  # Default if data not available

    def _calculate_quality(self, data):
        """Calculate product quality rate."""
        try:
            good_parts = data['good_parts'].sum()
            total_parts = data['total_parts'].sum()
            return good_parts / total_parts if total_parts > 0 else 0
        except:
            return 0.98  # Default if data not available

    def _calculate_mtbf(self, data):
        """Calculate Mean Time Between Failures."""
        try:
            total_uptime = data['uptime'].sum()
            num_failures = data['failures'].sum()
            return total_uptime / num_failures if num_failures > 0 else float('inf')
        except:
            return 168  # Default: 1 week in hours

    def _calculate_mttr(self, data):
        """Calculate Mean Time To Repair."""
        try:
            total_repair_time = data['repair_time'].sum()
            num_repairs = data['repairs'].sum()
            return total_repair_time / num_repairs if num_repairs > 0 else 0
        except:
            return 4  # Default: 4 hours
