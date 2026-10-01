import requests
payload = {"path":"data/example_article.txt","title":"Example Article","source":"example"}
r = requests.post("http://localhost:8000/ingest", json=payload)
print(r.json())
