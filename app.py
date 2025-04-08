from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
import os
from datetime import datetime, timedelta
import jwt
from functools import wraps
from models import db, User, Machine, Sensor, DataPoint, SensorReading, ModelMetrics
from data_collector import DataCollector
from ml_module import ContinuousMLTrainer
from data_simulator import DataSimulator

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-please-change')

# Database configuration
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # Handle Render's PostgreSQL URL format
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Fallback to SQLite for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///factorysync.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Configure SocketIO with appropriate CORS settings for Render
if os.environ.get('RENDER'):
    socketio = SocketIO(app, cors_allowed_origins=['https://factorysync-ai.onrender.com'])
else:
    socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize data simulator
simulator = DataSimulator(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    """Initialize the database and create test machines and sensors."""
    try:
        # Drop all tables and recreate them
        db.drop_all()
        db.create_all()
        
        # Create admin user
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@example.com')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Created admin user successfully")
        
        # Create machines with their sensors
        machines = [
            {
                'name': 'CNC Mill XR2000',
                'type': 'CNC Machine',
                'status': 'active',
                'sensors': [
                    {'name': 'Spindle Speed', 'type': 'RPM', 'unit': 'RPM', 'min_value': 0, 'max_value': 12000},
                    {'name': 'Cutting Force', 'type': 'Force', 'unit': 'N', 'min_value': 0, 'max_value': 5000},
                    {'name': 'Tool Temperature', 'type': 'Temperature', 'unit': '°C', 'min_value': 0, 'max_value': 200}
                ]
            },
            {
                'name': 'Injection Molder IM500',
                'type': 'Injection Molding',
                'status': 'active',
                'sensors': [
                    {'name': 'Mold Temperature', 'type': 'Temperature', 'unit': '°C', 'min_value': 0, 'max_value': 300},
                    {'name': 'Injection Pressure', 'type': 'Pressure', 'unit': 'bar', 'min_value': 0, 'max_value': 2000},
                    {'name': 'Screw Position', 'type': 'Position', 'unit': 'mm', 'min_value': 0, 'max_value': 100}
                ]
            },
            {
                'name': 'Robot Arm UR10',
                'type': 'Robotic Arm',
                'status': 'active',
                'sensors': [
                    {'name': 'Joint 1 Position', 'type': 'Angle', 'unit': 'degrees', 'min_value': -360, 'max_value': 360},
                    {'name': 'Joint 2 Position', 'type': 'Angle', 'unit': 'degrees', 'min_value': -360, 'max_value': 360},
                    {'name': 'End Effector Force', 'type': 'Force', 'unit': 'N', 'min_value': 0, 'max_value': 100}
                ]
            },
            {
                'name': 'Laser Cutter LC1000',
                'type': 'Laser Cutting',
                'status': 'active',
                'sensors': [
                    {'name': 'Laser Power', 'type': 'Power', 'unit': 'W', 'min_value': 0, 'max_value': 1000},
                    {'name': 'Cutting Speed', 'type': 'Speed', 'unit': 'mm/s', 'min_value': 0, 'max_value': 500},
                    {'name': 'Assist Gas Pressure', 'type': 'Pressure', 'unit': 'bar', 'min_value': 0, 'max_value': 25}
                ]
            },
            {
                'name': 'Assembly Line AL200',
                'type': 'Assembly Line',
                'status': 'active',
                'sensors': [
                    {'name': 'Line Speed', 'type': 'Speed', 'unit': 'units/hr', 'min_value': 0, 'max_value': 1000},
                    {'name': 'Motor Temperature', 'type': 'Temperature', 'unit': '°C', 'min_value': 0, 'max_value': 120},
                    {'name': 'Power Consumption', 'type': 'Power', 'unit': 'kW', 'min_value': 0, 'max_value': 50}
                ]
            },
            {
                'name': 'Heat Treatment Furnace HT600',
                'type': 'Heat Treatment',
                'status': 'active',
                'sensors': [
                    {'name': 'Chamber Temperature', 'type': 'Temperature', 'unit': '°C', 'min_value': 0, 'max_value': 1200},
                    {'name': 'Humidity', 'type': 'Humidity', 'unit': '%RH', 'min_value': 0, 'max_value': 100},
                    {'name': 'Gas Flow Rate', 'type': 'Flow', 'unit': 'L/min', 'min_value': 0, 'max_value': 100}
                ]
            }
        ]

        # Add machines and their sensors to the database
        for machine_data in machines:
            machine = Machine(
                name=machine_data['name'],
                type=machine_data['type'],
                status=machine_data['status']
            )
            db.session.add(machine)
            db.session.flush()  # Get the machine ID

            # Add sensors for this machine
            for sensor_data in machine_data['sensors']:
                sensor = Sensor(
                    machine_id=machine.id,
                    name=sensor_data['name'],
                    type=sensor_data['type'],
                    unit=sensor_data['unit'],
                    min_value=sensor_data['min_value'],
                    max_value=sensor_data['max_value']
                )
                db.session.add(sensor)

        db.session.commit()
        print("Created machines and sensors successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        db.session.rollback()
        raise

# Initialize database on startup
with app.app_context():
    init_db()

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    # Start the simulator when someone accesses the dashboard
    simulator.start()
    return render_template('dashboard.html')

@app.route('/stop-simulation', methods=['POST'])
@login_required
def stop_simulation():
    simulator.stop()
    return jsonify({'status': 'success', 'message': 'Simulation stopped'})

@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    user = User(
        username=data['username'],
        email=data.get('email'),
        role=data.get('role', 'user')
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login_api():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({'message': 'Invalid username or password'}), 401
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['SECRET_KEY'])
    
    return jsonify({'token': token})

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/api/machines', methods=['GET'])
@token_required
def get_machines(current_user):
    machines = Machine.query.all()
    return jsonify([{
        'id': m.id,
        'name': m.name,
        'type': m.type,
        'status': m.status,
        'created_at': m.created_at.isoformat()
    } for m in machines])

@app.route('/api/machines/<int:machine_id>/data', methods=['GET'])
@token_required
def get_machine_data(current_user, machine_id):
    machine = Machine.query.get_or_404(machine_id)
    data_points = DataPoint.query.filter_by(machine_id=machine_id).order_by(DataPoint.timestamp.desc()).limit(100).all()
    return jsonify([{
        'timestamp': dp.timestamp.isoformat(),
        'readings': [{
            'sensor': r.sensor.name,
            'value': r.value,
            'unit': r.sensor.unit
        } for r in dp.readings]
    } for dp in data_points])

@app.route('/api/machines/<int:machine_id>/metrics', methods=['GET'])
@token_required
def get_machine_metrics(current_user, machine_id):
    metrics = ModelMetrics.query.filter_by(machine_id=machine_id).order_by(ModelMetrics.timestamp.desc()).first()
    if not metrics:
        return jsonify({'message': 'No metrics available'}), 404
    return jsonify({
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1_score': metrics.f1_score,
        'timestamp': metrics.timestamp.isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    return redirect(url_for('login'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Health check endpoint for Render
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': str(e)
        }), 500

# Manual database initialization endpoint (protected)
@app.route('/init-db', methods=['POST'])
def manual_init_db():
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {os.environ.get('SECRET_KEY', 'dev-key-please-change')}":
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
    try:
        init_db()
        return jsonify({'status': 'success', 'message': 'Database initialized successfully'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_sensor_data')
def handle_sensor_data_request(data):
    """Handle real-time sensor data requests."""
    machine_id = data.get('machine_id')
    if machine_id:
        # Get the latest readings for the machine
        readings = SensorReading.query.join(Sensor).filter(
            Sensor.machine_id == machine_id
        ).order_by(SensorReading.timestamp.desc()).limit(10).all()
        
        # Format the data
        data = [{
            'sensor_name': reading.sensor.name,
            'value': reading.value,
            'timestamp': reading.timestamp.isoformat(),
            'anomaly_score': reading.anomaly_score
        } for reading in readings]
        
        socketio.emit('sensor_data_update', {
            'machine_id': machine_id,
            'readings': data
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    if os.environ.get('RENDER'):
        socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    else:
        socketio.run(app, host='0.0.0.0', port=port)
