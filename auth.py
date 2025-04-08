from functools import wraps
from flask import request, jsonify
from flask_login import LoginManager, current_user
import jwt
from datetime import datetime, timedelta
from models import User, db
import os

login_manager = LoginManager()

def init_auth(app):
    login_manager.init_app(app)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

def generate_token(user):
    """Generate JWT token for API authentication"""
    payload = {
        'user_id': user.id,
        'username': user.username,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, os.environ.get('SECRET_KEY', 'dev-key-change-in-production'), algorithm='HS256')

def token_required(f):
    """Decorator for API endpoints that require token authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        try:
            data = jwt.decode(token, os.environ.get('SECRET_KEY', 'dev-key-change-in-production'), algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid user'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

def admin_required(f):
    """Decorator for endpoints that require admin privileges"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated

def register_user(username, email, password, role='user'):
    """Register a new user"""
    if User.query.filter_by(username=username).first():
        return False, "Username already exists"
        
    if User.query.filter_by(email=email).first():
        return False, "Email already exists"
        
    user = User(username=username, email=email, role=role)
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        return True, "User registered successfully"
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def verify_user(username, password):
    """Verify user credentials"""
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    return None
