import requests
import yaml
import subprocess
import logging
import time
from datetime import datetime
from anomaly_detector import AnomalyDetector

# === Logging Setup ===
logging.basicConfig(filename='logs/agent.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Load Config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

PROMETHEUS_URL = config['prometheus_url']
CHECK_INTERVAL = config['check_interval_sec']

# === Initialize Detector ===
detector = AnomalyDetector()

# === Query Prometheus API ===
def query_prometheus(query):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        r.raise_for_status()
        result = r.json()['data']['result']
        if result and 'value' in result[0]:
            return [float(result[0]['value'][1])]
        else:
            return []
    except Exception as e:
        logging.error(f"Prometheus query failed for '{query}': {e}")
        return []

# === Heal Action ===
def trigger_heal(action):
    try:
        subprocess.run(["bash", f"heal_scripts/{action}"], check=True)
        logging.info(f"Executed heal script: {action}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Heal script failed: {e}")

# === Main Monitoring Loop ===
def monitor_loop():
    logging.info("Agent started monitoring loop.")
    while True:
        for rule in config['rules']:
            metric_values = query_prometheus(rule['query'])

            if metric_values:
                if detector.is_anomaly(metric_values):
                    logging.warning(f"Anomaly detected for rule: {rule['name']}")
                    trigger_heal(rule['action'])
                else:
                    logging.info(f"No anomaly for rule: {rule['name']}")
            else:
                logging.warning(f"No data received for rule: {rule['name']}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor_loop()