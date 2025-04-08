# FactorySync AI

A machine learning-powered monitoring system for predictive maintenance and performance analysis of industrial machinery.

## Features

- **Failure Prediction**: Uses machine learning to predict potential failures based on real-time sensor data
- **Maintenance Monitoring**: Tracks maintenance schedules and alerts for upcoming or overdue maintenance
- **Machine Activity Tracking**: Monitors machine state and current operations in real-time
- **Multi-Protocol Support**: Integrates with various data sources:
  - MTConnect for modern CNC machines
  - OPC UA for industrial control systems
  - IIoT APIs for cloud-connected machinery

## Supported Machines

- CNC Machines (e.g., Haas VF-2)
- Drilling Machines (e.g., Delta 18-900L)
- Grinding Machines (e.g., Okamoto ACC-818NC)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python init_db.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open http://localhost:5000 in your browser

## Configuration

- Machine data sources can be configured in the web interface
- Default maintenance intervals can be adjusted in the database
- Model parameters can be tuned in the ML pipeline

## Development

- Built with Flask and SQLAlchemy
- Uses Bootstrap 5 for the frontend
- Implements real-time monitoring with AJAX
- Supports Heroku deployment

## License

MIT License
