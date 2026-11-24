"""Test whether the topic details endpoint returns the full text of a grant call."""

import requests

CALL_ID = "HORIZON-CL5-2023-D4-02-02"
URL = f"https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{CALL_ID.lower()}.json"

response = requests.get(URL, timeout=30)
print("status code:", response.status_code)

if response.ok:
    data = response.json()
    print("top-level keys:", list(data.keys()))
    details = data.get("TopicDetails", data)
    print("detail keys:", list(details.keys())[:30])
    for key, value in details.items():
        if isinstance(value, str) and len(value) > 500:
            print(f"  {key}: {len(value)} chars -> {value[:150]!r}")
