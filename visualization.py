import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from models import Machine, Sensor, SensorReading, DataPoint
from sqlalchemy import func
import json

class DashboardVisualizer:
    def __init__(self):
        self.color_scheme = {
            'healthy': '#2ecc71',
            'warning': '#f1c40f',
            'critical': '#e74c3c',
            'unknown': '#95a5a6'
        }

    def create_machine_overview(self, machine_id, timeframe_hours=24):
        """Create an overview visualization for a specific machine."""
        try:
            # Get machine data
            since = datetime.utcnow() - timedelta(hours=timeframe_hours)
            readings = SensorReading.query.join(Sensor).filter(
                Sensor.machine_id == machine_id,
                SensorReading.timestamp > since
            ).all()
            
            if not readings:
                return None

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'sensor_name': r.sensor.name,
                'value': r.value,
                'anomaly_score': r.anomaly_score,
                'failure_probability': r.failure_probability
            } for r in readings])

            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Sensor Values', 'Anomaly Scores', 'Failure Probability'),
                vertical_spacing=0.1
            )

            # Plot sensor values
            for sensor in df['sensor_name'].unique():
                sensor_data = df[df['sensor_name'] == sensor]
                fig.add_trace(
                    go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data['value'],
                        name=sensor,
                        mode='lines'
                    ),
                    row=1, col=1
                )

            # Plot anomaly scores
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['anomaly_score'],
                    name='Anomaly Score',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )

            # Plot failure probability
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['failure_probability'],
                    name='Failure Probability',
                    line=dict(color='red')
                ),
                row=3, col=1
            )

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text=f"Machine Overview - Past {timeframe_hours} Hours"
            )

            return fig.to_json()

        except Exception as e:
            print(f"Error creating machine overview: {str(e)}")
            return None

    def create_oee_dashboard(self, machine_id, timeframe_hours=24):
        """Create OEE (Overall Equipment Effectiveness) visualization."""
        try:
            # Get OEE data
            since = datetime.utcnow() - timedelta(hours=timeframe_hours)
            kpis = DataPoint.query.filter(
                DataPoint.machine_id == machine_id,
                DataPoint.timestamp > since,
                DataPoint.type.like('kpi_%')
            ).all()

            if not kpis:
                return None

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': k.timestamp,
                'type': k.type.replace('kpi_', ''),
                'value': float(k.value)
            } for k in kpis])

            # Create gauge charts for current OEE components
            latest = df.loc[df.groupby('type')['timestamp'].idxmax()]
            
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "scatter"}]
                ],
                subplot_titles=("OEE", "Availability", "Performance", "Trend")
            )

            # OEE Gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=latest[latest['type'] == 'oee']['value'].iloc[0] * 100,
                    title={'text': "OEE"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': self.color_scheme['healthy']}}
                ),
                row=1, col=1
            )

            # Availability Gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=latest[latest['type'] == 'availability']['value'].iloc[0] * 100,
                    title={'text': "Availability"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': self.color_scheme['healthy']}}
                ),
                row=1, col=2
            )

            # Performance Gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=latest[latest['type'] == 'performance']['value'].iloc[0] * 100,
                    title={'text': "Performance"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': self.color_scheme['healthy']}}
                ),
                row=2, col=1
            )

            # OEE Trend
            oee_trend = df[df['type'] == 'oee'].sort_values('timestamp')
            fig.add_trace(
                go.Scatter(
                    x=oee_trend['timestamp'],
                    y=oee_trend['value'] * 100,
                    name='OEE Trend',
                    line=dict(color=self.color_scheme['healthy'])
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text=f"OEE Dashboard - Past {timeframe_hours} Hours"
            )

            return fig.to_json()

        except Exception as e:
            print(f"Error creating OEE dashboard: {str(e)}")
            return None

    def create_maintenance_schedule(self, machine_id):
        """Create maintenance schedule visualization."""
        try:
            # Get maintenance recommendations
            recommendations = DataPoint.query.filter(
                DataPoint.machine_id == machine_id,
                DataPoint.type == 'maintenance_recommendation'
            ).order_by(DataPoint.timestamp.desc()).limit(10).all()

            if not recommendations:
                return None

            # Parse recommendations
            tasks = []
            for rec in recommendations:
                data = json.loads(rec.value)
                tasks.append({
                    'Task': data['maintenance_type'][0],
                    'Start': data['recommended_window']['start'],
                    'End': data['recommended_window']['end'],
                    'Priority': data['priority']
                })

            df = pd.DataFrame(tasks)

            # Create Gantt chart
            fig = px.timeline(
                df, 
                x_start="Start",
                x_end="End",
                y="Task",
                color="Priority",
                color_discrete_map={
                    'HIGH': self.color_scheme['critical'],
                    'MEDIUM': self.color_scheme['warning']
                }
            )

            fig.update_layout(
                title_text="Maintenance Schedule",
                height=400,
                showlegend=True
            )

            return fig.to_json()

        except Exception as e:
            print(f"Error creating maintenance schedule: {str(e)}")
            return None

    def create_health_status_card(self, health_status):
        """Create a health status card visualization."""
        try:
            fig = go.Figure()

            # Add health score gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=health_status['health_score'] * 100,
                title={'text': f"Health Status: {health_status['status'].title()}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self.color_scheme[health_status['status']]},
                    'steps': [
                        {'range': [0, 50], 'color': self.color_scheme['critical']},
                        {'range': [50, 80], 'color': self.color_scheme['warning']},
                        {'range': [80, 100], 'color': self.color_scheme['healthy']}
                    ]
                }
            ))

            # Add alerts as annotations
            for i, alert in enumerate(health_status['alerts']):
                fig.add_annotation(
                    text=alert,
                    xref="paper", yref="paper",
                    x=0, y=-0.2 - (i * 0.1),
                    showarrow=False,
                    font=dict(
                        color=self.color_scheme['critical'] if 'critical' in alert.lower() 
                        else self.color_scheme['warning']
                    )
                )

            fig.update_layout(
                height=300,
                margin=dict(t=80, b=100)  # Adjust margins to fit alerts
            )

            return fig.to_json()

        except Exception as e:
            print(f"Error creating health status card: {str(e)}")
            return None

    def create_factory_overview(self):
        """Create a factory-wide overview visualization."""
        try:
            # Get all machines
            machines = Machine.query.all()
            
            # Collect latest readings and status for each machine
            data = []
            for machine in machines:
                latest_reading = SensorReading.query.join(Sensor).filter(
                    Sensor.machine_id == machine.id
                ).order_by(SensorReading.timestamp.desc()).first()
                
                if latest_reading:
                    health_score = 1 - (latest_reading.failure_probability * 0.7 + 
                                     latest_reading.anomaly_score * 0.3)
                    status = ('healthy' if health_score > 0.8 else 
                             'warning' if health_score > 0.5 else 'critical')
                else:
                    health_score = 0
                    status = 'unknown'
                
                data.append({
                    'machine_name': machine.name,
                    'type': machine.type,
                    'status': status,
                    'health_score': health_score * 100
                })
            
            df = pd.DataFrame(data)
            
            # Create treemap
            fig = px.treemap(
                df,
                path=[px.Constant("Factory"), 'type', 'machine_name'],
                values='health_score',
                color='status',
                color_discrete_map=self.color_scheme
            )
            
            fig.update_layout(
                title_text="Factory Health Overview",
                height=600
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating factory overview: {str(e)}")
            return None

    def create_pareto_chart(self, machine_id, timeframe_hours=168):
        """Create a Pareto chart of issues/anomalies."""
        try:
            # Get anomaly data
            since = datetime.utcnow() - timedelta(hours=timeframe_hours)
            readings = SensorReading.query.join(Sensor).filter(
                Sensor.machine_id == machine_id,
                SensorReading.timestamp > since,
                SensorReading.anomaly_score > 0.8  # Only significant anomalies
            ).all()
            
            if not readings:
                return None
            
            # Group by sensor and count anomalies
            anomalies = pd.DataFrame([{
                'sensor_name': r.sensor.name,
                'count': 1
            } for r in readings])
            
            anomalies = anomalies.groupby('sensor_name')['count'].sum().reset_index()
            anomalies = anomalies.sort_values('count', ascending=False)
            
            # Calculate cumulative percentage
            total = anomalies['count'].sum()
            anomalies['cumulative_percent'] = (anomalies['count'].cumsum() / total) * 100
            
            # Create Pareto chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bars
            fig.add_trace(
                go.Bar(
                    x=anomalies['sensor_name'],
                    y=anomalies['count'],
                    name="Anomaly Count"
                ),
                secondary_y=False
            )
            
            # Add line
            fig.add_trace(
                go.Scatter(
                    x=anomalies['sensor_name'],
                    y=anomalies['cumulative_percent'],
                    name="Cumulative %",
                    line=dict(color='red')
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title_text=f"Pareto Chart of Anomalies - Past {timeframe_hours} Hours",
                height=400,
                showlegend=True,
                xaxis_title="Sensor",
                yaxis_title="Number of Anomalies",
                yaxis2_title="Cumulative Percentage"
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating Pareto chart: {str(e)}")
            return None
