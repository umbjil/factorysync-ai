from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Machine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    type = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='idle')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    config = db.Column(db.JSON)
    
    sensors = db.relationship('Sensor', backref='machine', lazy=True)
    data_points = db.relationship('DataPoint', backref='machine', lazy=True)

class Sensor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    unit = db.Column(db.String(20), nullable=False)
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    warning_min = db.Column(db.Float)
    warning_max = db.Column(db.Float)
    
    readings = db.relationship('SensorReading', backref='sensor', lazy=True)

class DataPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_failure = db.Column(db.Boolean, default=False)
    prediction_confidence = db.Column(db.Float)
    is_anomaly = db.Column(db.Boolean, default=False)
    
    readings = db.relationship('SensorReading', backref='data_point', lazy=True)

class SensorReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data_point_id = db.Column(db.Integer, db.ForeignKey('data_point.id'), nullable=False)
    sensor_id = db.Column(db.Integer, db.ForeignKey('sensor.id'), nullable=False)
    value = db.Column(db.Float, nullable=False)

class ModelMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    samples_processed = db.Column(db.Integer)
    
    @property
    def as_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'samples_processed': self.samples_processed
        }
