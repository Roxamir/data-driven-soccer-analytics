# Citations:
# [1] https://www.api-football.com/news/post/how-to-get-all-teams-and-players-from-a-league-id
# [2] https://www.api-football.com/documentation#players-statistics-seasons


import requests
import time
import json
import pandas as pd

def call_api(league_id, season, page):

	headers = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': "MY-API-KEY" # I have one, but I paid for it, so I do not want to share it.
    }
	
	url = 'https://v3.football.api-sports.io/players'

	query_string = {"league" : str(league_id), "season" : str(season), "page" : str(page)}
	
	response = requests.get(
        url,
        headers=headers,
        params=query_string
    )

	return response.json()

def get_player_data(league_id, season, page, minimum_appearances):
    # First, read existing data (if any)
    try:
        with open(f'players_data_{league_id}_{season}.json', 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Filtered data list to store players meeting minimum appearances
    filtered_data = []

    # Iterate through all pages
    while True:
        # Get players from the current page
        players = call_api(league_id, season, page)
        
        # Filter players based on minimum appearances
        for player_entry in players.get('response', []):
            # Check if the player has statistics and meets minimum appearances
            if player_entry.get('statistics') and len(player_entry['statistics']) > 0:
                # Safely get appearances, defaulting to 0 if None
                appearances = player_entry['statistics'][0].get('games', {}).get('appearences') or 0
                
                # Convert to int to handle potential string values
                try:
                    appearances = int(appearances)
                except (ValueError, TypeError):
                    appearances = 0
                
                if appearances >= minimum_appearances:
                    filtered_data.append(player_entry)
        
        # Print current page and total pages for tracking
        total_pages = players['paging']['total']
        print(f"Processed page {page} of {total_pages}")
        
        # Check if we've reached the last page
        if page >= total_pages:
            print("Reached last page")
            break
        
        # Move to next page
        page += 1
        
        # Small delay to avoid hitting API rate limits
        time.sleep(1)

    # Write filtered data back to the file
    with open(f'players_data_{league_id}_{season}_min{minimum_appearances}.json', 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Finished collecting player data with minimum {minimum_appearances} appearances.")
    print(f"Total players meeting criteria: {len(filtered_data)}")

# Big 5 leagues (England, Germany, France, Italy, Spain)
league_ids = [39, 61, 78, 135, 140]
seasons = [2020, 2021, 2022, 2023, 2024]

# Get players with at least 10 appearances
for league_id in league_ids:
    for season in seasons:
        page = 1
        get_player_data(league_id, season, page, minimum_appearances=10)