#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order
"""

import requests

def fetch_data(url):
    """Fetch JSON data from a given URL."""
    response = requests.get(url)
    response.raise_for_status()  # Ensures the request was successful
    return response.json()

if __name__ == "__main__":
    # Fetch all launches and rockets once to minimize API calls
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'

    launches = fetch_data(launches_url)
    rockets_data = {rocket["id"]: rocket["name"] for rocket in fetch_data(rockets_url)}

    # Count launches per rocket
    rocket_launch_count = {}
    for launch in launches:
        rocket_name = rockets_data.get(launch.get('rocket'), "Unknown Rocket")
        rocket_launch_count[rocket_name] = rocket_launch_count.get(rocket_name, 0) + 1

    # Sort: First by number of launches (descending), then alphabetically
    sorted_rockets = sorted(rocket_launch_count.items(), key=lambda x: (-x[1], x[0]))

    # Print results
    for rocket, count in sorted_rockets:
        print(f"{rocket}: {count}")

