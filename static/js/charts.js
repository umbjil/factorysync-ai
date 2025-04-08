class ChartManager {
    constructor() {
        this.charts = new Map();
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            layout: {
                padding: {
                    left: 10,
                    right: 10,
                    top: 20,
                    bottom: 10
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true
                }
            }
        };
    }

    createTimeSeriesChart(containerId, data, options = {}) {
        const chartOptions = {
            ...this.defaultOptions,
            ...options,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour',
                        displayFormats: {
                            hour: 'HH:mm',
                            day: 'MMM D'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: {
                        display: true,
                        drawBorder: true,
                        drawOnChartArea: true,
                        drawTicks: true
                    }
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: options.yAxisLabel || ''
                    },
                    grid: {
                        display: true,
                        drawBorder: true,
                        drawOnChartArea: true,
                        drawTicks: true
                    }
                }
            }
        };

        const config = {
            type: 'line',
            data: {
                datasets: [{
                    label: data.label,
                    data: data.values,
                    borderColor: data.color || '#0d6efd',
                    backgroundColor: this.addAlpha(data.color || '#0d6efd', 0.1),
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0.4
                }]
            },
            options: chartOptions
        };

        const chart = new Chart(document.getElementById(containerId), config);
        this.charts.set(containerId, chart);
        return chart;
    }

    updateTimeSeriesChart(containerId, newData) {
        const chart = this.charts.get(containerId);
        if (chart) {
            chart.data.datasets[0].data = newData;
            chart.update('none');
        }
    }

    createGaugeChart(containerId, value, options = {}) {
        const config = {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [value, 100 - value],
                    backgroundColor: [
                        this.getColorForValue(value),
                        '#e9ecef'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                ...this.defaultOptions,
                circumference: 180,
                rotation: -90,
                cutout: '80%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        };

        const chart = new Chart(document.getElementById(containerId), config);
        this.charts.set(containerId, chart);
        return chart;
    }

    createMultiLineChart(containerId, datasets, options = {}) {
        const config = {
            type: 'line',
            data: {
                labels: options.labels || [],
                datasets: datasets.map(dataset => ({
                    label: dataset.label,
                    data: dataset.values,
                    borderColor: dataset.color,
                    backgroundColor: this.addAlpha(dataset.color, 0.1),
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0.4
                }))
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    x: {
                        type: options.xAxisType || 'category',
                        title: {
                            display: true,
                            text: options.xAxisLabel || ''
                        }
                    },
                    y: {
                        beginAtZero: options.beginAtZero || false,
                        title: {
                            display: true,
                            text: options.yAxisLabel || ''
                        }
                    }
                }
            }
        };

        const chart = new Chart(document.getElementById(containerId), config);
        this.charts.set(containerId, chart);
        return chart;
    }

    createBarChart(containerId, data, options = {}) {
        const config = {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: data.label,
                    data: data.values,
                    backgroundColor: data.colors || Array(data.values.length).fill('#0d6efd'),
                    borderWidth: 1
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: options.yAxisLabel || ''
                        }
                    }
                }
            }
        };

        const chart = new Chart(document.getElementById(containerId), config);
        this.charts.set(containerId, chart);
        return chart;
    }

    destroyChart(containerId) {
        const chart = this.charts.get(containerId);
        if (chart) {
            chart.destroy();
            this.charts.delete(containerId);
        }
    }

    addAlpha(color, alpha) {
        const rgb = this.hexToRgb(color);
        return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    getColorForValue(value) {
        if (value >= 80) return '#198754';  // success
        if (value >= 60) return '#0dcaf0';  // info
        if (value >= 40) return '#ffc107';  // warning
        return '#dc3545';  // danger
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chartManager = new ChartManager();

    // Initialize charts if on machine details page
    if (document.querySelector('.machine-details')) {
        // Temperature chart
        const temperatureData = {
            label: 'Temperature (K)',
            values: JSON.parse(document.getElementById('temperature-data').dataset.values),
            color: '#dc3545'
        };
        chartManager.createTimeSeriesChart('temperature-chart', temperatureData, {
            yAxisLabel: 'Temperature (K)'
        });

        // RPM chart
        const rpmData = {
            label: 'RPM',
            values: JSON.parse(document.getElementById('rpm-data').dataset.values),
            color: '#0d6efd'
        };
        chartManager.createTimeSeriesChart('rpm-chart', rpmData, {
            yAxisLabel: 'RPM'
        });

        // Tool wear chart
        const toolWearData = {
            label: 'Tool Wear (min)',
            values: JSON.parse(document.getElementById('tool-wear-data').dataset.values),
            color: '#ffc107'
        };
        chartManager.createTimeSeriesChart('tool-wear-chart', toolWearData, {
            yAxisLabel: 'Tool Wear (min)'
        });
    }

    // Initialize charts if on dashboard page
    if (document.querySelector('.dashboard-content')) {
        // Machine status distribution
        const statusData = {
            labels: ['Running', 'Idle', 'Stopped'],
            label: 'Machines',
            values: JSON.parse(document.getElementById('status-distribution').dataset.values),
            colors: ['#198754', '#ffc107', '#dc3545']
        };
        chartManager.createBarChart('status-chart', statusData);

        // Maintenance schedule
        const maintenanceData = JSON.parse(document.getElementById('maintenance-schedule').dataset.values);
        chartManager.createMultiLineChart('maintenance-chart', maintenanceData.datasets, {
            labels: maintenanceData.labels,
            xAxisLabel: 'Next 7 Days',
            yAxisLabel: 'Machines',
            beginAtZero: true
        });
    }
});
