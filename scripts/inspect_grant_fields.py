"""Find which API field holds the text of a grant call."""

import json

import requests

API_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

params = {"apiKey": "SEDIA", "text": "flood", "pageSize": 1, "pageNumber": 1}
query = {
    "bool": {
        "must": [
            {"terms": {"type": ["1", "2"]}},  # grants only
            {"terms": {"status": ["31094501", "31094502"]}},  # forthcoming, open
        ]
    }
}
files = {
    "query": ("blob", json.dumps(query), "application/json"),
    "languages": ("blob", json.dumps(["en"]), "application/json"),
}

result = requests.post(API_URL, params=params, files=files, timeout=30).json()["results"][0]
metadata = result["metadata"]

print("identifier:", metadata.get("identifier"))
print("top-level summary length:", len(result.get("summary") or ""))
print("top-level content length:", len(result.get("content") or ""))
print("metadata fields with long text:")
for key, value in metadata.items():
    text = " ".join(value) if isinstance(value, list) else str(value)
    if len(text) > 200:
        print(f"  {key}: {len(text)} chars -> {text[:120]!r}")
