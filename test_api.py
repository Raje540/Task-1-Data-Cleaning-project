import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "hours_studied": 5,
    "attendance": 80,
    "previous_score": 70
}

response = requests.post(url, json=data)
print("STATUS:", response.status_code)
print("TEXT:", response.text)
