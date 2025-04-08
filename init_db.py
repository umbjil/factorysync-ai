from app import app, db, User
from werkzeug.security import generate_password_hash
from datetime import datetime

def init_db():
    with app.app_context():
        print("Creating database tables...")
        db.drop_all()
        db.create_all()
        
        print("\nCreating admin user...")
        admin = User(
            username='admin',
            email='admin@factorysync.ai',
            password_hash=generate_password_hash('admin123'),
            role='admin'
        )
        
        print("Adding admin to database...")
        db.session.add(admin)
        db.session.commit()
        
        print("\nVerifying admin user...")
        admin_check = User.query.filter_by(username='admin').first()
        if admin_check:
            print("[OK] Admin user created successfully!")
            print(f"[OK] Username: {admin_check.username}")
            print(f"[OK] Email: {admin_check.email}")
            print(f"[OK] Role: {admin_check.role}")
            print("\nUse these credentials to log in:")
            print("Username: admin")
            print("Password: admin123")
        else:
            print("[ERROR] Admin user was not created!")

if __name__ == '__main__':
    init_db()
