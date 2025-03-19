#!/usr/bin/env python3
"""
Provides statistics about Nginx logs stored in MongoDB.
- Counts total logs
- Shows request method statistics (GET, POST, etc.)
- Counts status check requests (GET to path /status)
"""
from pymongo import MongoClient

def log_stats():
    """
    Fetches and prints log statistics from MongoDB's 'logs.nginx' collection.
    Statistics include:
    - Total log count
    - Count per HTTP method (GET/POST/PUT/PATCH/DELETE)
    - Count of status check requests (GET to /status)
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client.logs
    collection = db.nginx

    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_check = collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print(f"{status_check} status check")

    client.close()

if __name__ == "__main__":
    log_stats()
