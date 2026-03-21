"""
Comprehensive Monitoring System for Speculative Decoding
Provides real-time metrics, health checks, and Prometheus-compatible endpoints
"""

import os
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
from flask import Flask, Response, jsonify
import psutil
import torch

# Prometheus client library
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Metrics export will be limited.")

@dataclass
class MetricSnapshot:
    """Single point-in-time metric snapshot"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PercentileMetrics:
    """Percentile calculations for latency metrics"""
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    count: int

class MetricsCollector:
    """Collects and aggregates metrics for monitoring"""
    
    def __init__(self, window_size_seconds: int = 300):  # 5-minute window
        self.window_size = window_size_seconds
        self.metrics_lock = threading.Lock()
        
        # Time series data storage
        self.throughput_samples = deque(maxlen=1000)
        self.latency_samples = defaultdict(lambda: deque(maxlen=1000))
        self.acceptance_rates = defaultdict(lambda: deque(maxlen=1000))
        self.memory_samples = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.model_selection_counts = defaultdict(int)
        self.fallback_counts = defaultdict(int)
        
        # Current values for gauges
        self.current_memory_gb = 0.0
        self.current_gpu_memory_gb = 0.0
        self.current_cpu_percent = 0.0
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
            
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Counters
        self.prom_inference_total = Counter(
            'speculative_decoding_inference_total',
            'Total number of inference requests',
            ['model_mode', 'status']
        )
        self.prom_tokens_generated = Counter(
            'speculative_decoding_tokens_generated_total',
            'Total tokens generated',
            ['model_mode']
        )
        self.prom_errors_total = Counter(
            'speculative_decoding_errors_total',
            'Total errors',
            ['error_type']
        )
        self.prom_fallback_total = Counter(
            'speculative_decoding_fallback_total',
            'Total fallback activations',
            ['from_mode', 'to_mode']
        )
        
        # Gauges
        self.prom_memory_usage_gb = Gauge(
            'speculative_decoding_memory_usage_gb',
            'Current memory usage in GB',
            ['memory_type']
        )
        self.prom_acceptance_rate = Gauge(
            'speculative_decoding_acceptance_rate',
            'Current acceptance rate',
            ['stage', 'model_mode']
        )
        self.prom_cpu_usage_percent = Gauge(
            'speculative_decoding_cpu_usage_percent',
            'Current CPU usage percentage'
        )
        
        # Histograms
        self.prom_latency_seconds = Histogram(
            'speculative_decoding_latency_seconds',
            'Request latency in seconds',
            ['operation', 'model_mode'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )
        self.prom_throughput_tps = Histogram(
            'speculative_decoding_throughput_tokens_per_second',
            'Throughput in tokens per second',
            ['model_mode'],
            buckets=(10, 50, 100, 200, 500, 1000, 2000)
        )
        
    def record_inference(self, 
                        duration_seconds: float,
                        tokens_generated: int,
                        model_mode: str,
                        success: bool,
                        acceptance_metrics: Optional[Dict[str, Any]] = None):
        """Record inference metrics"""
        with self.metrics_lock:
            timestamp = time.time()
            
            # Calculate throughput
            throughput = tokens_generated / duration_seconds if duration_seconds > 0 else 0
            self.throughput_samples.append(MetricSnapshot(timestamp, throughput, {"mode": model_mode}))
            
            # Record latency
            self.latency_samples[model_mode].append(MetricSnapshot(timestamp, duration_seconds))
            
            # Record acceptance rates if provided
            if acceptance_metrics:
                if "stage1_acceptance_rate" in acceptance_metrics:
                    self.acceptance_rates["stage1"].append(
                        MetricSnapshot(timestamp, acceptance_metrics["stage1_acceptance_rate"])
                    )
                if "stage2_acceptance_rate" in acceptance_metrics:
                    self.acceptance_rates["stage2"].append(
                        MetricSnapshot(timestamp, acceptance_metrics["stage2_acceptance_rate"])
                    )
                if "combined_acceptance_rate" in acceptance_metrics:
                    self.acceptance_rates["combined"].append(
                        MetricSnapshot(timestamp, acceptance_metrics["combined_acceptance_rate"])
                    )
                    
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                status_label = "success" if success else "failure"
                self.prom_inference_total.labels(model_mode=model_mode, status=status_label).inc()
                self.prom_tokens_generated.labels(model_mode=model_mode).inc(tokens_generated)
                self.prom_latency_seconds.labels(operation="inference", model_mode=model_mode).observe(duration_seconds)
                self.prom_throughput_tps.labels(model_mode=model_mode).observe(throughput)
                
                if acceptance_metrics:
                    for stage, rate_key in [
                        ("stage1", "stage1_acceptance_rate"),
                        ("stage2", "stage2_acceptance_rate"),
                        ("combined", "combined_acceptance_rate")
                    ]:
                        if rate_key in acceptance_metrics:
                            self.prom_acceptance_rate.labels(
                                stage=stage, model_mode=model_mode
                            ).set(acceptance_metrics[rate_key])
                            
    def record_memory(self, memory_gb: float, gpu_memory_gb: float, cpu_percent: float):
        """Record memory usage"""
        with self.metrics_lock:
            timestamp = time.time()
            self.memory_samples.append(MetricSnapshot(timestamp, memory_gb + gpu_memory_gb))
            
            self.current_memory_gb = memory_gb
            self.current_gpu_memory_gb = gpu_memory_gb
            self.current_cpu_percent = cpu_percent
            
            if PROMETHEUS_AVAILABLE:
                self.prom_memory_usage_gb.labels(memory_type="system").set(memory_gb)
                self.prom_memory_usage_gb.labels(memory_type="gpu").set(gpu_memory_gb)
                self.prom_cpu_usage_percent.set(cpu_percent)
                
    def record_error(self, error_type: str):
        """Record error occurrence"""
        with self.metrics_lock:
            self.error_counts[error_type] += 1
            
            if PROMETHEUS_AVAILABLE:
                self.prom_errors_total.labels(error_type=error_type).inc()
                
    def record_fallback(self, from_mode: str, to_mode: str):
        """Record fallback activation"""
        with self.metrics_lock:
            fallback_key = f"{from_mode}_to_{to_mode}"
            self.fallback_counts[fallback_key] += 1
            
            if PROMETHEUS_AVAILABLE:
                self.prom_fallback_total.labels(from_mode=from_mode, to_mode=to_mode).inc()
                
    def record_model_selection(self, model_mode: str):
        """Record model selection for request"""
        with self.metrics_lock:
            self.model_selection_counts[model_mode] += 1
            
    def get_percentile_metrics(self, samples: deque, window_seconds: Optional[int] = None) -> PercentileMetrics:
        """Calculate percentile metrics for a time series"""
        if not samples:
            return PercentileMetrics(0, 0, 0, 0, 0, 0, 0)
            
        current_time = time.time()
        window_start = current_time - (window_seconds or self.window_size)
        
        # Filter samples within window
        values = [s.value for s in samples if s.timestamp >= window_start]
        
        if not values:
            return PercentileMetrics(0, 0, 0, 0, 0, 0, 0)
            
        sorted_values = sorted(values)
        count = len(values)
        
        return PercentileMetrics(
            p50=sorted_values[int(count * 0.5)],
            p95=sorted_values[int(count * 0.95)],
            p99=sorted_values[int(count * 0.99)],
            mean=statistics.mean(values),
            min=min(values),
            max=max(values),
            count=count
        )
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.metrics_lock:
            # Calculate throughput percentiles
            throughput_metrics = self.get_percentile_metrics(self.throughput_samples)
            
            # Calculate latency percentiles by model mode
            latency_by_mode = {}
            for mode, samples in self.latency_samples.items():
                latency_by_mode[mode] = asdict(self.get_percentile_metrics(samples))
                
            # Calculate acceptance rate averages
            acceptance_rates = {}
            for stage, samples in self.acceptance_rates.items():
                recent_samples = [s.value for s in samples if s.timestamp >= time.time() - 300]
                acceptance_rates[stage] = {
                    "current": recent_samples[-1] if recent_samples else 0,
                    "average": statistics.mean(recent_samples) if recent_samples else 0
                }
                
            # Get recent memory peak
            recent_memory = [s.value for s in self.memory_samples if s.timestamp >= time.time() - 300]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "throughput_tokens_per_second": asdict(throughput_metrics),
                "latency_seconds_by_mode": latency_by_mode,
                "acceptance_rates": acceptance_rates,
                "memory_usage_gb": {
                    "current": self.current_memory_gb + self.current_gpu_memory_gb,
                    "peak": max(recent_memory) if recent_memory else 0,
                    "system": self.current_memory_gb,
                    "gpu": self.current_gpu_memory_gb
                },
                "cpu_usage_percent": self.current_cpu_percent,
                "error_counts": dict(self.error_counts),
                "model_selection_frequency": dict(self.model_selection_counts),
                "fallback_activation_counts": dict(self.fallback_counts)
            }
            
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        with self.metrics_lock:
            # Check latency alerts
            for mode, samples in self.latency_samples.items():
                metrics = self.get_percentile_metrics(samples, window_seconds=60)
                if metrics.p95 > 5.0:  # 5 second threshold
                    alerts.append({
                        "type": "high_latency",
                        "severity": "warning",
                        "message": f"High latency detected for {mode}: p95={metrics.p95:.2f}s",
                        "value": metrics.p95,
                        "threshold": 5.0
                    })
                    
            # Check acceptance rate alerts
            for stage, samples in self.acceptance_rates.items():
                recent_samples = [s.value for s in samples if s.timestamp >= time.time() - 300]
                if recent_samples:
                    avg_rate = statistics.mean(recent_samples)
                    if avg_rate < 0.6:  # 60% threshold
                        alerts.append({
                            "type": "low_acceptance_rate",
                            "severity": "warning",
                            "message": f"Low acceptance rate for {stage}: {avg_rate:.2%}",
                            "value": avg_rate,
                            "threshold": 0.6
                        })
                        
            # Check memory alerts
            total_memory = self.current_memory_gb + self.current_gpu_memory_gb
            if total_memory > 11.5:  # 11.5GB threshold
                alerts.append({
                    "type": "high_memory_usage",
                    "severity": "critical" if total_memory > 12.0 else "warning",
                    "message": f"High memory usage: {total_memory:.1f}GB",
                    "value": total_memory,
                    "threshold": 11.5
                })
                
            # Check error rate alerts
            total_errors = sum(self.error_counts.values())
            if total_errors > 10:  # More than 10 errors
                error_rate = total_errors / max(sum(self.model_selection_counts.values()), 1)
                if error_rate > 0.01:  # 1% error rate
                    alerts.append({
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": f"High error rate: {error_rate:.2%}",
                        "value": error_rate,
                        "threshold": 0.01
                    })
                    
        return alerts


class MonitoringServer:
    """Flask server for monitoring endpoints"""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8080):
        self.app = Flask(__name__)
        self.metrics_collector = metrics_collector
        self.port = port
        self.start_time = time.time()
        
        # Configure routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/health')
        def health():
            """Basic liveness check"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        @self.app.route('/ready')
        def ready():
            """Readiness check"""
            # Check if system is ready to serve traffic
            alerts = self.metrics_collector.check_alerts()
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            
            if critical_alerts:
                return jsonify({
                    "status": "not_ready",
                    "reason": "critical alerts active",
                    "alerts": critical_alerts
                }), 503
                
            return jsonify({
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        @self.app.route('/metrics')
        def metrics():
            """Prometheus-compatible metrics endpoint"""
            if PROMETHEUS_AVAILABLE:
                return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
            else:
                # Fallback to JSON metrics
                return jsonify(self.metrics_collector.get_metrics_summary())
                
        @self.app.route('/debug')
        def debug():
            """Detailed diagnostics endpoint"""
            # Get system information
            cpu_info = {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }
            
            memory_info = psutil.virtual_memory()._asdict()
            
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
                }
            else:
                gpu_info = {"available": False}
                
            return jsonify({
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "system": {
                    "cpu": cpu_info,
                    "memory": memory_info,
                    "gpu": gpu_info
                },
                "metrics": self.metrics_collector.get_metrics_summary(),
                "alerts": self.metrics_collector.check_alerts()
            })
            
        @self.app.route('/alerts')
        def alerts():
            """Active alerts endpoint"""
            active_alerts = self.metrics_collector.check_alerts()
            return jsonify({
                "timestamp": datetime.utcnow().isoformat(),
                "alert_count": len(active_alerts),
                "alerts": active_alerts
            })
            
    def run(self, host: str = '0.0.0.0', threaded: bool = True):
        """Run the monitoring server"""
        self.app.run(host=host, port=self.port, threaded=threaded)


class BackgroundMonitor(threading.Thread):
    """Background thread for continuous monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector, interval_seconds: int = 10):
        super().__init__(daemon=True)
        self.metrics_collector = metrics_collector
        self.interval = interval_seconds
        self.running = True
        
    def run(self):
        """Monitor system resources in background"""
        while self.running:
            try:
                # Get memory usage
                mem = psutil.virtual_memory()
                memory_gb = (mem.total - mem.available) / (1024**3)
                
                # Get GPU memory if available
                gpu_memory_gb = 0
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.memory_reserved() / (1024**3)
                    
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Record metrics
                self.metrics_collector.record_memory(memory_gb, gpu_memory_gb, cpu_percent)
                
                # Check alerts
                alerts = self.metrics_collector.check_alerts()
                for alert in alerts:
                    if alert["severity"] == "critical":
                        print(f"CRITICAL ALERT: {alert['message']}")
                        
            except Exception as e:
                print(f"Error in background monitor: {e}")
                
            time.sleep(self.interval)
            
    def stop(self):
        """Stop background monitoring"""
        self.running = False


# Convenience function to start monitoring
def start_monitoring(port: int = 8080) -> Tuple[MetricsCollector, MonitoringServer, BackgroundMonitor]:
    """Start the complete monitoring system"""
    # Create metrics collector
    collector = MetricsCollector()
    
    # Create and start background monitor
    background_monitor = BackgroundMonitor(collector)
    background_monitor.start()
    
    # Create monitoring server
    server = MonitoringServer(collector, port)
    
    return collector, server, background_monitor


# Export main components
__all__ = [
    'MetricsCollector',
    'MonitoringServer',
    'BackgroundMonitor',
    'start_monitoring',
    'MetricSnapshot',
    'PercentileMetrics'
]