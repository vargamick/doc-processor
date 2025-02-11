from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import threading
from .validation import EmbeddingValidator, ValidationConfig
from .base import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class QualityMetricsConfig:
    """Configuration for quality metrics collection and analysis"""

    validation_window: int = 1000  # Number of embeddings to consider for trend analysis
    min_quality_score: float = 0.8  # Minimum acceptable quality score
    trend_threshold: float = -0.05  # Threshold for negative trend detection
    metrics_log_path: str = "metrics_log.jsonl"
    validation_interval: timedelta = timedelta(hours=1)
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class QualityMetrics:
    """Collects and analyzes embedding quality metrics"""

    def __init__(
        self,
        validator: EmbeddingValidator,
        config: Optional[QualityMetricsConfig] = None,
    ):
        self.validator = validator
        self.config = config or QualityMetricsConfig()
        self.metrics_history: List[Dict[str, Any]] = []
        self.last_validation_time = datetime.now()
        self._lock = threading.Lock()

    def calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results"""
        try:
            # Extract component scores
            dimension_score = float(
                validation_results["dimension_validation"]["consistent"]
            )
            semantic_score = float(
                validation_results["semantic_validation"]["mean_similarity"]
            )
            outlier_score = (
                1.0 - validation_results["outlier_detection"]["outlier_ratio"]
                if "outlier_ratio" in validation_results["outlier_detection"]
                else 1.0
            )

            # Weighted combination of scores
            weights = {
                "dimension": 0.4,
                "semantic": 0.4,
                "outlier": 0.2,
            }

            quality_score = (
                weights["dimension"] * dimension_score
                + weights["semantic"] * semantic_score
                + weights["outlier"] * outlier_score
            )

            return quality_score
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in quality metrics"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}

        try:
            # Extract quality scores from history
            scores = [m["quality_score"] for m in self.metrics_history]
            times = [
                datetime.fromisoformat(m["timestamp"]) for m in self.metrics_history
            ]

            # Calculate trend statistics
            score_changes = np.diff(scores)
            avg_change = float(np.mean(score_changes))
            std_change = float(np.std(score_changes))

            # Detect significant trends
            trend_detected = avg_change < self.config.trend_threshold
            current_score = scores[-1]
            below_threshold = current_score < self.config.min_quality_score

            analysis = {
                "status": "alert" if trend_detected or below_threshold else "normal",
                "current_score": current_score,
                "average_change": avg_change,
                "std_deviation": std_change,
                "trend_detected": trend_detected,
                "below_threshold": below_threshold,
                "window_size": len(scores),
                "time_span": str(times[-1] - times[0]),
            }

            if trend_detected or below_threshold:
                self._send_alert(
                    "Quality metric alert",
                    {
                        "type": "trend_alert" if trend_detected else "threshold_alert",
                        "analysis": analysis,
                    },
                )

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _send_alert(self, message: str, data: Dict[str, Any]) -> None:
        """Send alert through configured callback"""
        if self.config.alert_callback:
            try:
                self.config.alert_callback(message, data)
            except Exception as e:
                logger.error(f"Error sending alert: {str(e)}")

    def log_metrics(self, validation_results: Dict[str, Any]) -> None:
        """Log validation results and calculated metrics"""
        try:
            with self._lock:
                quality_score = self.calculate_quality_score(validation_results)
                metric_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "quality_score": quality_score,
                    "validation_results": validation_results,
                }

                # Add to in-memory history
                self.metrics_history.append(metric_entry)
                # Keep only recent history
                if len(self.metrics_history) > self.config.validation_window:
                    self.metrics_history.pop(0)

                # Write to log file
                log_path = Path(self.config.metrics_log_path)
                with log_path.open("a") as f:
                    f.write(json.dumps(metric_entry) + "\n")

        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics and analysis"""
        if not self.metrics_history:
            return {"status": "no_data"}

        try:
            latest = self.metrics_history[-1]
            trend_analysis = self.analyze_trends()

            return {
                "latest_score": latest["quality_score"],
                "timestamp": latest["timestamp"],
                "trend_analysis": trend_analysis,
                "validation_results": latest["validation_results"],
            }
        except Exception as e:
            logger.error(f"Error getting latest metrics: {str(e)}")
            return {"status": "error", "message": str(e)}


class ModelValidation:
    """Manages periodic model validation and quality monitoring"""

    def __init__(
        self,
        processor: EmbeddingProcessor,
        validation_config: Optional[ValidationConfig] = None,
        metrics_config: Optional[QualityMetricsConfig] = None,
    ):
        self.processor = processor
        self.validator = EmbeddingValidator(processor, validation_config)
        self.metrics = QualityMetrics(self.validator, metrics_config)
        self._validation_thread = None
        self._stop_validation = threading.Event()

    def start_periodic_validation(self) -> None:
        """Start periodic validation in a background thread"""
        if self._validation_thread and self._validation_thread.is_alive():
            return

        self._stop_validation.clear()
        self._validation_thread = threading.Thread(
            target=self._run_periodic_validation, daemon=True
        )
        self._validation_thread.start()

    def stop_periodic_validation(self) -> None:
        """Stop periodic validation"""
        if self._validation_thread:
            self._stop_validation.set()
            self._validation_thread.join()

    def _run_periodic_validation(self) -> None:
        """Run validation periodically"""
        while not self._stop_validation.is_set():
            try:
                if (
                    datetime.now() - self.metrics.last_validation_time
                    >= self.metrics.config.validation_interval
                ):
                    self._validate_model()
                    self.metrics.last_validation_time = datetime.now()

                # Sleep for a short interval before checking again
                self._stop_validation.wait(timeout=60)
            except Exception as e:
                logger.error(f"Error in periodic validation: {str(e)}")
                # Sleep before retrying
                self._stop_validation.wait(timeout=300)

    def _validate_model(self) -> None:
        """Perform model validation and log results"""
        try:
            # Generate test embeddings using reference texts
            test_embeddings = [
                self.processor.generate_embedding(text)
                for text in self.validator.config.reference_texts
            ]

            # Run validation
            valid, results = self.validator.validate_batch(test_embeddings)

            # Log metrics
            self.metrics.log_metrics(results)

            if not valid:
                self._handle_validation_failure(results)

        except Exception as e:
            logger.error(f"Model validation error: {str(e)}")
            self._handle_validation_failure({"error": str(e)})

    def _handle_validation_failure(self, results: Dict[str, Any]) -> None:
        """Handle validation failures"""
        try:
            # Get trend analysis
            trend_analysis = self.metrics.analyze_trends()

            # Prepare alert data
            alert_data = {
                "validation_results": results,
                "trend_analysis": trend_analysis,
                "model_info": self.processor.get_model_info(),
            }

            # Send alert
            self.metrics._send_alert("Model validation failed", alert_data)

        except Exception as e:
            logger.error(f"Error handling validation failure: {str(e)}")

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status and metrics"""
        try:
            return {
                "metrics": self.metrics.get_latest_metrics(),
                "validator_stats": self.validator.get_validation_stats(),
                "model_info": self.processor.get_model_info(),
            }
        except Exception as e:
            logger.error(f"Error getting validation status: {str(e)}")
            return {"status": "error", "message": str(e)}
