import os
from datetime import timedelta

class Config:
    # Basic Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_123')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), "instance", "factorysync.db"))}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Email configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@factorysync.ai')
    
    # Cache configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # API configuration
    API_TITLE = 'FactorySync AI API'
    API_VERSION = 'v1'
    API_DESCRIPTION = 'RESTful API for FactorySync AI'
    
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour;1 per second"
    RATELIMIT_STORAGE_URL = "memory://"
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    
    # Machine learning model settings
    MODEL_UPDATE_INTERVAL = 3600  # seconds
    PREDICTION_THRESHOLD = 0.5
    
    # Monitoring settings
    ALERT_CHECK_INTERVAL = 300  # seconds
    MAINTENANCE_REMINDER_DAYS = 7
    DATA_RETENTION_DAYS = 90
    
    # Machine types
    MACHINE_TYPES = {
        'L': {
            'name': 'CNC Machine',
            'model': 'Haas VF-2',
            'maintenance_interval': 500,  # hours
            'parameters': {
                'max_rpm': 8000,
                'max_torque': 100,
                'max_tool_wear': 200,
                'normal_temp_range': (290, 320)
            }
        },
        'M': {
            'name': 'Drilling Machine',
            'model': 'Delta 18-900L',
            'maintenance_interval': 400,
            'parameters': {
                'max_rpm': 3000,
                'max_torque': 50,
                'max_tool_wear': 150,
                'normal_temp_range': (285, 310)
            }
        },
        'H': {
            'name': 'Grinding Machine',
            'model': 'Okamoto ACC-818NC',
            'maintenance_interval': 600,
            'parameters': {
                'max_rpm': 10000,
                'max_torque': 30,
                'max_tool_wear': 250,
                'normal_temp_range': (295, 325)
            }
        }
    }
    
    # Data source configurations
    DATA_SOURCES = {
        'mtconnect': {
            'protocol': 'http',
            'default_port': 5000,
            'endpoints': {
                'current': '/current',
                'sample': '/sample',
                'asset': '/asset',
                'probe': '/probe'
            }
        },
        'opcua': {
            'protocol': 'opc.tcp',
            'default_port': 4840,
            'security_mode': 'None',
            'security_policy': 'None'
        },
        'iiot': {
            'protocol': 'https',
            'default_port': 443,
            'auth_type': 'bearer',
            'endpoints': {
                'data': '/api/v1/data',
                'status': '/api/v1/status',
                'control': '/api/v1/control'
            }
        }
    }
    
    # User roles and permissions
    ROLES = {
        'admin': {
            'name': 'Administrator',
            'permissions': ['read', 'write', 'delete', 'manage_users', 'manage_machines', 'api_access']
        },
        'manager': {
            'name': 'Manager',
            'permissions': ['read', 'write', 'manage_machines']
        },
        'operator': {
            'name': 'Operator',
            'permissions': ['read', 'write']
        },
        'viewer': {
            'name': 'Viewer',
            'permissions': ['read']
        }
    }
    
    # Alert settings
    ALERT_TYPES = {
        'maintenance_due': {
            'severity': 'WARNING',
            'message': 'Maintenance is due in {days} days'
        },
        'high_temperature': {
            'severity': 'WARNING',
            'message': 'Temperature exceeds normal range'
        },
        'failure_predicted': {
            'severity': 'ERROR',
            'message': 'High probability of failure detected'
        },
        'tool_wear': {
            'severity': 'WARNING',
            'message': 'Tool wear approaching limit'
        },
        'system_error': {
            'severity': 'CRITICAL',
            'message': 'System error detected'
        }
    }
