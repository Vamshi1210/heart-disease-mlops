"""
Send realistic test traffic to the API so the Grafana dashboard has something to show.

Sends random (valid) patients, plus about 1 in 10 deliberately invalid requests.
Run:  python monitoring/send_traffic.py http://<API-ADDRESS> 300
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request


def random_patient() -> dict:
    return {
        "age": random.randint(29, 77),
        "sex": random.choice([0, 1]),
        "chest_pain_type": random.choice([1, 2, 3, 4]),
        "resting_bp": random.randint(94, 200),
        "cholesterol": random.randint(126, 400),
        "fasting_blood_sugar": random.choice([0, 1]),
        "resting_ecg": random.choice([0, 2]),
        "max_heart_rate": random.randint(90, 200),
        "exercise_angina": random.choice([0, 1]),
        "st_depression": round(random.uniform(0, 4), 1),
        "st_slope": random.choice([1, 2, 3]),
        "num_major_vessels": random.choice([0, 0, 1, 2, 3]),
        "thalassemia": random.choice([3, 6, 7]),
    }


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    statuses = {}
    for i in range(1, count + 1):
        patient = random_patient()
        if random.random() < 0.1:
            patient["age"] = -5  # invalid on purpose: should be rejected with 422
        code = post(f"{base}/predict", patient)
        statuses[code] = statuses.get(code, 0) + 1
        if i % 25 == 0:
            print(f"{i}/{count} sent, status counts so far: {statuses}")
        time.sleep(0.3)
    print(f"Done. Status counts: {statuses}")


if __name__ == "__main__":
    main()
