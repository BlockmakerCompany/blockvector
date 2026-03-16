import urllib.request
import json
import random

# Generamos 1536 floats aleatorios
query_vector = [random.uniform(-1.0, 1.0) for _ in range(1536)]

data = {
    "top_k": 5,
    "query": query_vector
}

req = urllib.request.Request("http://localhost:8080/search", method="POST")
req.add_header('Content-Type', 'application/json')

response = urllib.request.urlopen(req, data=json.dumps(data).encode('utf-8'))
print(response.read().decode('utf-8'))