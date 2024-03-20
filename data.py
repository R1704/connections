import requests
from bs4 import BeautifulSoup
import json
import os

n_games = 281

# URL of the page to scrape
base_url = 'https://connections.swellgarfo.com/nyt/'

for i in range(1, n_games+1):

    url = os.path.join(base_url, str(i))

    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the <script> tag containing the data
    script_tag = soup.find('script', id='__NEXT_DATA__')

    # Extract the JSON string and parse it
    data_json = json.loads(script_tag.text)

    # Construct a filename based on the URL or another unique identifier
    filename = f'game_data_{i}.json'
    filepath = os.path.join('data', filename)  # Specify your save directory

    # Save the JSON data to a file
    with open(filepath, 'w') as file:
        json.dump(data_json, file)
