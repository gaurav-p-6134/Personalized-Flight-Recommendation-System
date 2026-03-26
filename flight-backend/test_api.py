import requests
import json

# This mimics the JSON your React frontend will send
mock_payload = {
    "ranker_id": 999,
    "flights": [
        {
            "Id": "flight_expensive_but_fast",
            "totalPrice": 85000,
            "taxes": 5000,
            "legs0_duration": "0.02:30:00", # 2.5 hours
            "legs0_segments0_marketingCarrier_code": "SU",
            "legs0_segments0_cabinClass": 2,
            "searchRoute": "MOWLED"
        },
        {
            "Id": "flight_cheap_but_slow",
            "totalPrice": 25000,
            "taxes": 3000,
            "legs0_duration": "0.15:45:00", # ~16 hours
            "legs0_segments0_marketingCarrier_code": "S7",
            "legs0_segments0_cabinClass": 3,
            "searchRoute": "MOWLED"
        }
    ]
}

# Send the POST request to your local server
print("Sending flights to the ranking API...")
response = requests.post("http://127.0.0.1:8000/recommend", json=mock_payload)

# Print the results
if response.status_code == 200:
    print("Success! Ranked results:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}: {response.text}")