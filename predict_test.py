import requests

url = 'http://localhost:9696/predict'

input_data = {
    "datetime": "2013-01-17 15:00:00",
    "season": 1,
    "holiday": 0,
    "workingday": 1,
    "weather": 1,
    "temp": 10.21,
    "atemp": 8.32,
    "humidity": 77,
    "windspeed": 5.9987,
    # Note: We don't need 'atemp' as it's not used in your preprocess
    # Note: We don't send 'year' or 'hour', as preprocess() *creates* them
}

response = requests.post(url, json=input_data).json()
print(response)