// Real-time updates
class RealTimeUpdater {
    constructor(interval = 30000) {
        this.interval = interval;
        this.updateHandlers = new Map();
        this.running = false;
    }

    addHandler(key, handler) {
        this.updateHandlers.set(key, handler);
    }

    removeHandler(key) {
        this.updateHandlers.delete(key);
    }

    start() {
        if (!this.running) {
            this.running = true;
            this.update();
            this.timer = setInterval(() => this.update(), this.interval);
        }
    }

    stop() {
        if (this.running) {
            this.running = false;
            clearInterval(this.timer);
        }
    }

    async update() {
        try {
            const response = await fetch(window.location.pathname + '/data');
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            this.updateHandlers.forEach(handler => handler(data));
        } catch (error) {
            console.error('Update failed:', error);
        }
    }
}

// Alert management
class AlertManager {
    constructor() {
        this.bindEvents();
    }

    bindEvents() {
        document.addEventListener('click', event => {
            if (event.target.matches('.acknowledge-alert')) {
                this.acknowledgeAlert(event.target);
            }
        });
    }

    async acknowledgeAlert(button) {
        const alertId = button.dataset.alertId;
        try {
            const response = await fetch(`/alert/${alertId}/acknowledge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const alertElement = button.closest('.alert-item');
            alertElement.classList.add('fade-out');
            setTimeout(() => alertElement.remove(), 500);
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
            showToast('Error acknowledging alert', 'error');
        }
    }
}

// Toast notifications
class ToastManager {
    constructor() {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.className = 'position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(this.container);
        }
    }

    show(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast show bg-${type}`;
        toast.innerHTML = `
            <div class="toast-header">
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body text-white">
                ${message}
            </div>
        `;

        this.container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 500);
        }, duration);
    }
}

// Form validation
class FormValidator {
    constructor(form) {
        this.form = form;
        this.bindEvents();
    }

    bindEvents() {
        this.form.addEventListener('submit', event => {
            if (!this.validateForm()) {
                event.preventDefault();
                event.stopPropagation();
            }
            this.form.classList.add('was-validated');
        });
    }

    validateForm() {
        let isValid = true;
        const inputs = this.form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            if (input.hasAttribute('required') && !input.value) {
                isValid = false;
                this.showError(input, 'This field is required');
            } else if (input.type === 'email' && !this.validateEmail(input.value)) {
                isValid = false;
                this.showError(input, 'Please enter a valid email address');
            }
        });

        return isValid;
    }

    validateEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    showError(input, message) {
        const feedback = input.nextElementSibling;
        if (feedback && feedback.classList.contains('invalid-feedback')) {
            feedback.textContent = message;
        } else {
            const div = document.createElement('div');
            div.className = 'invalid-feedback';
            div.textContent = message;
            input.parentNode.insertBefore(div, input.nextSibling);
        }
    }
}

// Initialize components
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));

    // Initialize popovers
    const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
    popovers.forEach(popover => new bootstrap.Popover(popover));

    // Initialize form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => new FormValidator(form));

    // Initialize alert manager
    const alertManager = new AlertManager();

    // Initialize toast manager
    window.toastManager = new ToastManager();

    // Initialize real-time updates if on dashboard or machine details page
    if (document.querySelector('.dashboard-content') || document.querySelector('.machine-details')) {
        const updater = new RealTimeUpdater();
        
        // Add handlers for different components
        updater.addHandler('machineStatus', data => {
            updateMachineStatus(data.machineStatus);
        });
        
        updater.addHandler('alerts', data => {
            updateAlerts(data.alerts);
        });
        
        updater.addHandler('predictions', data => {
            updatePredictions(data.predictions);
        });
        
        updater.start();
    }
});

// Utility functions
function updateMachineStatus(status) {
    const statusElements = document.querySelectorAll('[data-machine-status]');
    statusElements.forEach(element => {
        const machineId = element.dataset.machineId;
        if (status[machineId]) {
            element.innerHTML = status[machineId];
            element.classList.add('real-time-update');
            setTimeout(() => element.classList.remove('real-time-update'), 1000);
        }
    });
}

function updateAlerts(alerts) {
    const alertContainer = document.getElementById('alerts-container');
    if (alertContainer && alerts.length > 0) {
        const alertsHtml = alerts.map(alert => `
            <div class="alert-item mb-3">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="alert-severity ${alert.severity.toLowerCase()}"></span>
                        ${alert.message}
                    </div>
                    <button class="btn btn-sm btn-outline-secondary acknowledge-alert" 
                            data-alert-id="${alert.id}">
                        Acknowledge
                    </button>
                </div>
            </div>
        `).join('');
        
        alertContainer.innerHTML = alertsHtml;
    }
}

function updatePredictions(predictions) {
    const predictionContainer = document.getElementById('predictions-container');
    if (predictionContainer && predictions.length > 0) {
        const predictionsHtml = predictions.map(prediction => `
            <div class="prediction-item mb-3">
                <div class="d-flex justify-content-between align-items-center">
                    <span>${prediction.type}</span>
                    <small>${new Date(prediction.timestamp).toLocaleString()}</small>
                </div>
                <div class="progress prediction-progress">
                    <div class="progress-bar bg-${prediction.probability > 0.7 ? 'danger' : 
                                                  prediction.probability > 0.3 ? 'warning' : 'success'}"
                         role="progressbar"
                         style="width: ${prediction.probability * 100}%">
                        ${Math.round(prediction.probability * 100)}%
                    </div>
                </div>
            </div>
        `).join('');
        
        predictionContainer.innerHTML = predictionsHtml;
    }
}

function showToast(message, type = 'info') {
    if (window.toastManager) {
        window.toastManager.show(message, type);
    }
}
