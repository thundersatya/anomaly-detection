import pytest
from anomaly_detector import AnomalyDetector

def test_zscore_detects_spike():
    detector = AnomalyDetector(zscore_threshold=2.0)
    values = [1.0, 1.1, 1.0, 1.2, 3.5]  # Last value is a spike
    assert detector.detect_zscore_spike(values) is True

def test_zscore_no_spike():
    detector = AnomalyDetector(zscore_threshold=3.0)
    values = [1.0, 1.1, 1.0, 1.2, 1.05]  # No spike
    assert detector.detect_zscore_spike(values) is False

def test_ewma_detects_drift():
    detector = AnomalyDetector(ewma_alpha=0.3, ewma_tolerance=1.1)
    values = [1.0, 1.05, 1.08, 1.10, 1.25]  # Drift upward
    assert detector.detect_ewma_drift(values) is True

def test_ewma_no_drift():
    detector = AnomalyDetector(ewma_alpha=0.3, ewma_tolerance=1.25)
    values = [1.0, 1.05, 1.08, 1.10, 1.15]  # Slight increase, but within tolerance
    assert detector.detect_ewma_drift(values) is False

def test_isolation_forest_detects_outlier():
    detector = AnomalyDetector()
    data = [
        [0.2, 0.3],
        [0.21, 0.29],
        [0.19, 0.31],
        [1.0, 2.5],  # clear outlier
    ]
    detector.train_isolation_forest(data[:-1])
    assert detector.detect_isolation_forest(data[-1]) is True

def test_isolation_forest_normal():
    detector = AnomalyDetector()
    data = [
        [0.2, 0.3],
        [0.21, 0.29],
        [0.19, 0.31],
    ]
    detector.train_isolation_forest(data)
    assert detector.detect_isolation_forest([0.22, 0.32]) is False