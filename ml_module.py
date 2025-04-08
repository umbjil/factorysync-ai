from datetime import datetime
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from machine_config import MachineConfig

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
