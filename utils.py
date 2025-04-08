import json
from datetime import datetime, timedelta
from functools import wraps
from flask import current_app, request, jsonify
from flask_login import current_user
import jwt
import requests
from opcua import Client
import numpy as np
from models import db, Machine, MachineData, Alert, Prediction, MaintenanceLog
import plotly.graph_objects as go
import plotly.utils
from config import Config

def create_alert(machine_id, alert_type, message, severity=None, data=None):
    """Create a new alert for a machine"""
    if severity is None:
        severity = Config.ALERT_TYPES.get(alert_type, {}).get('severity', 'INFO')
    
    alert = Alert(
        machine_id=machine_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        data=json.dumps(data) if data else None
    )
    db.session.add(alert)
    db.session.commit()
    return alert

def check_maintenance_status(machine):
    """Check maintenance status and create alerts if needed"""
    now = datetime.utcnow()
    next_maintenance = machine.last_maintenance + timedelta(hours=machine.maintenance_interval)
    days_until = (next_maintenance - now).days
    
    if days_until <= 0:
        create_alert(
            machine.id,
            'maintenance_due',
            f'Maintenance overdue by {abs(days_until)} days',
            'ERROR'
        )
    elif days_until <= Config.MAINTENANCE_REMINDER_DAYS:
        create_alert(
            machine.id,
            'maintenance_due',
            f'Maintenance due in {days_until} days',
            'WARNING'
        )
    return next_maintenance, days_until

def fetch_machine_data(machine):
    """Fetch real-time data from the machine using its configured data source"""
    if machine.data_source == 'mtconnect':
        return fetch_mtconnect_data(machine)
    elif machine.data_source == 'opcua':
        return fetch_opcua_data(machine)
    elif machine.data_source == 'iiot':
        return fetch_iiot_data(machine)
    return None

def fetch_mtconnect_data(machine):
    """Fetch data from MTConnect endpoint"""
    try:
        url = f"http://{machine.ip_address}:5000/current"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return process_mtconnect_data(data)
    except Exception as e:
        create_alert(machine.id, 'system_error', f'Failed to fetch MTConnect data: {str(e)}')
    return None

def fetch_opcua_data(machine):
    """Fetch data from OPC UA server"""
    try:
        client = Client(f"opc.tcp://{machine.ip_address}:4840")
        client.connect()
        try:
            data = {
                'air_temp': client.get_node("ns=2;s=AirTemperature").get_value(),
                'process_temp': client.get_node("ns=2;s=ProcessTemperature").get_value(),
                'rpm': client.get_node("ns=2;s=RPM").get_value(),
                'torque': client.get_node("ns=2;s=Torque").get_value(),
                'tool_wear': client.get_node("ns=2;s=ToolWear").get_value(),
            }
            return data
        finally:
            client.disconnect()
    except Exception as e:
        create_alert(machine.id, 'system_error', f'Failed to fetch OPC UA data: {str(e)}')
    return None

def fetch_iiot_data(machine):
    """Fetch data from IIoT API"""
    try:
        config = machine.get_config()
        headers = {'Authorization': f'Bearer {config.get("api_key")}'}
        url = f"https://{machine.ip_address}/api/v1/data"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        create_alert(machine.id, 'system_error', f'Failed to fetch IIoT data: {str(e)}')
    return None

def process_machine_data(machine, data):
    """Process and store machine data, create alerts if needed"""
    if not data:
        return None
        
    # Store the data point
    data_point = MachineData(
        machine_id=machine.id,
        timestamp=datetime.utcnow(),
        air_temp=data.get('air_temp'),
        process_temp=data.get('process_temp'),
        rpm=data.get('rpm'),
        torque=data.get('torque'),
        tool_wear=data.get('tool_wear'),
        machine_state=data.get('state'),
        operation=data.get('operation'),
        power_consumption=data.get('power'),
        vibration=data.get('vibration'),
        noise_level=data.get('noise'),
        coolant_pressure=data.get('coolant_pressure'),
        coolant_temperature=data.get('coolant_temp'),
        raw_data=json.dumps(data)
    )
    db.session.add(data_point)
    
    # Check for anomalies
    machine_config = Config.MACHINE_TYPES[machine.type]
    params = machine_config['parameters']
    
    if data.get('process_temp') > params['normal_temp_range'][1]:
        create_alert(
            machine.id,
            'high_temperature',
            f'Process temperature ({data["process_temp"]}K) exceeds normal range'
        )
    
    if data.get('tool_wear', 0) > params['max_tool_wear'] * 0.9:
        create_alert(
            machine.id,
            'tool_wear',
            f'Tool wear ({data["tool_wear"]} min) approaching maximum'
        )
    
    db.session.commit()
    return data_point

def generate_prediction(machine, data):
    """Generate failure prediction for a machine"""
    try:
        # Prepare data for the model
        X = np.array([[
            {'L': 0, 'M': 1, 'H': 2}[machine.type],
            data.air_temp,
            data.process_temp,
            data.rpm,
            data.torque,
            data.tool_wear
        ]])
        
        # Make prediction
        with current_app.app_context():
            probability = current_app.model.predict_proba(X)[0][1]
            
        prediction = Prediction(
            machine_id=machine.id,
            timestamp=datetime.utcnow(),
            prediction_type='failure',
            probability=probability,
            details=json.dumps({
                'features': {
                    'type': machine.type,
                    'air_temp': data.air_temp,
                    'process_temp': data.process_temp,
                    'rpm': data.rpm,
                    'torque': data.torque,
                    'tool_wear': data.tool_wear
                }
            })
        )
        db.session.add(prediction)
        
        if probability > Config.PREDICTION_THRESHOLD:
            create_alert(
                machine.id,
                'failure_predicted',
                f'High failure probability detected: {probability:.1%}'
            )
        
        db.session.commit()
        return prediction
    except Exception as e:
        create_alert(machine.id, 'system_error', f'Failed to generate prediction: {str(e)}')
        return None

def generate_machine_chart(machine_id, metric, timeframe='24h'):
    """Generate a Plotly chart for machine metrics"""
    now = datetime.utcnow()
    if timeframe == '24h':
        start_time = now - timedelta(hours=24)
    elif timeframe == '7d':
        start_time = now - timedelta(days=7)
    elif timeframe == '30d':
        start_time = now - timedelta(days=30)
    
    data_points = MachineData.query.filter(
        MachineData.machine_id == machine_id,
        MachineData.timestamp >= start_time
    ).order_by(MachineData.timestamp).all()
    
    x = [point.timestamp for point in data_points]
    y = [getattr(point, metric) for point in data_points]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=metric))
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} over time',
        xaxis_title='Time',
        yaxis_title=metric.replace('_', ' ').title(),
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def requires_permission(permission):
    """Decorator to check if user has required permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({'error': 'Authentication required'}), 401
            
            user_role = current_user.role
            if permission not in Config.ROLES[user_role]['permissions']:
                return jsonify({'error': 'Permission denied'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_api_key():
    """Validate API key from request header"""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return None
        
    from models import APIKey
    key = APIKey.query.filter_by(key=api_key, is_active=True).first()
    if not key:
        return None
        
    if key.expires_at and key.expires_at < datetime.utcnow():
        return None
        
    key.last_used = datetime.utcnow()
    db.session.commit()
    
    return key.user

def generate_api_key():
    """Generate a new API key"""
    return jwt.encode(
        {
            'created_at': datetime.utcnow().isoformat(),
            'random': np.random.randint(0, 1000000)
        },
        current_app.config['SECRET_KEY'],
        algorithm='HS256'
    )
