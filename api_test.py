import requests

url = "http://127.0.0.1:8000/api/v1/hackrx/run"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer 13e8f1be06f08115af28397a9f74cd278d9fc81b65945ec8a1c19a2c45511a4c"
}
payload = {
    "documents": "http://localhost:8080/sample.pdf",
    "questions": ["What is the maximum room rent limit per day as per this document?"]
}

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.json())
