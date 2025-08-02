import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, zscore_threshold=3.0, ewma_alpha=0.3, ewma_tolerance=1.25):
        self.zscore_threshold = zscore_threshold
        self.ewma_alpha = ewma_alpha
        self.ewma_tolerance = ewma_tolerance
        self.iforest_model = None

    def detect_zscore_spike(self, values):
        if len(values) < 5:
            return False
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return False
        z = abs(values[-1] - mean) / std
        return z > self.zscore_threshold

    def ewma(self, values):
        smoothed = values[0]
        for v in values[1:]:
            smoothed = self.ewma_alpha * v + (1 - self.ewma_alpha) * smoothed
        return smoothed

    def detect_ewma_drift(self, values):
        if len(values) < 5:
            return False
        smoothed = self.ewma(values[:-1])
        return values[-1] > (smoothed * self.ewma_tolerance)

    def train_isolation_forest(self, data_matrix):
        self.iforest_model = IsolationForest(n_estimators=50, contamination=0.05)
        self.iforest_model.fit(data_matrix)

    def detect_isolation_forest(self, last_row):
        if self.iforest_model is None:
            return False
        prediction = self.iforest_model.predict([last_row])[0]
        return prediction == -1  # anomaly

    def is_anomaly(self, values):
        return (
            self.detect_zscore_spike(values) or
            self.detect_ewma_drift(values)
        )