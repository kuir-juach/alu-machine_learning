#!/usr/bin/env python3
"""Displays the upcoming SpaceX launch details in the required format."""

import requests
from datetime import datetime

def fetch_data(url):
    """Fetch JSON data from the given URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    return response.json()

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = fetch_data(url)

    # Find the next launch based on the earliest date_unix
    next_launch = min(launches, key=lambda x: x["date_unix"])

    # Extract relevant details
    launch_name = next_launch["name"]
    date = next_launch["date_local"]  # Already formatted in expected format
    rocket_id = next_launch["rocket"]
    launchpad_id = next_launch["launchpad"]

    # Fetch rocket details
    rocket_name = fetch_data(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")["name"]

    # Fetch launchpad details
    launchpad_data = fetch_data(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    # Final output format
    output = f"{launch_name} ({date}) {rocket_name} - {launchpad_name} ({launchpad_locality})"
    
    print(output)

