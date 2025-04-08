from flask_socketio import SocketIO
from flask_login import current_user
from datetime import datetime

socketio = SocketIO(async_mode='gevent')

def init_websocket(app):
    socketio.init_app(app, async_mode='gevent', cors_allowed_origins="*")
    
    @socketio.on('connect')
    def handle_connect():
        if not current_user.is_authenticated:
            return False
        emit('connected', {'status': 'connected'})
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to machine updates"""
        if not current_user.is_authenticated:
            return
            
        machine_id = data.get('machine_id')
        if machine_id:
            join_room(f'machine_{machine_id}')
            emit('subscribed', {
                'machine_id': machine_id,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from machine updates"""
        machine_id = data.get('machine_id')
        if machine_id:
            leave_room(f'machine_{machine_id}')
            emit('unsubscribed', {
                'machine_id': machine_id,
                'timestamp': datetime.utcnow().isoformat()
            })

def broadcast_machine_update(machine_id, data):
    """Broadcast machine update to all clients"""
    socketio.emit('machine_update', {
        'machine_id': machine_id,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    })

def broadcast_alert(machine_id, alert_type, message, severity='info'):
    """Broadcast alert to all clients"""
    socketio.emit('alert', {
        'machine_id': machine_id,
        'type': alert_type,
        'message': message,
        'severity': severity,
        'timestamp': datetime.utcnow().isoformat()
    })

def broadcast_prediction(machine_id, prediction):
    """Broadcast prediction update to all clients"""
    socketio.emit('prediction', {
        'machine_id': machine_id,
        'prediction': prediction,
        'timestamp': datetime.utcnow().isoformat()
    })
