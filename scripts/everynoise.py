import requests
from bs4 import BeautifulSoup
import networkx as nx
import numpy as np
import json
import requests
import pickle
import os
import concurrent.futures
from bs4 import BeautifulSoup
import networkx as nx
import pandas as pd
from node2vec import Node2Vec

"""
This script scrapes genre relationships from Every Noise and trains node2vec embeddings on them
Essentially, we're creating a 'map' of the genres and their relationships to each other

The script performs the following steps:
1. Scrape genre relationships from the main page of Every Noise at Once
2. Train node2vec embeddings on the genre graph
3. Save the genre embeddings to a JSON file
    
Ensure that the node2vec library is installed before running this script.
"""

EVERY_NOISE_URL = "https://everynoise.com/"
CACHE_FILE = "genre_graph.pkl"

def scrape_genres():
    response = requests.get(EVERY_NOISE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    genres = {}

    # Extract all genres and their URLs
    for div in soup.find_all("div", class_="genre"):
        genre_name = div.text.strip()
        link = div.find("a")
        if link:
            genres[genre_name] = EVERY_NOISE_URL + link['href']

    return genres

# Scrape the genres and save them
genre_dict = scrape_genres()
print(f"Scraped {len(genre_dict)} genres")

def scrape_genre_relationships_main_page(genre_dict):
    """
    Extracts genre relationships directly from the main page of Every Noise at Once.
    This avoids scraping each individual genre's subpage.
    """
    G = nx.Graph()

    # Fetch the main page again
    response = requests.get(EVERY_NOISE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all genres from the main page
    genre_positions = []
    
    for div in soup.find_all("div", class_="genre"):
        genre_name = div.text.strip()
        if genre_name in genre_dict:
            # Extract the position (x, y) of the genre in the HTML layout
            style = div.get("style", "")
            position = tuple(map(float, [s.split(":")[1].replace("px", "") for s in style.split(";") if "left" in s or "top" in s]))
            genre_positions.append((genre_name, position))
            print(f"Processed genre: {genre_name} at position: {position}")

    # Convert to a dictionary for easy lookup
    genre_positions = dict(genre_positions)

    # Compute genre proximity based on their positions
    for genre1, pos1 in genre_positions.items():
        for genre2, pos2 in genre_positions.items():
            if genre1 != genre2:
                # Calculate distance between genres
                distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                
                # Only add an edge if genres are "close"
                if distance < 150:  # May need to tune
                    G.add_edge(genre1, genre2, weight=1 / (distance + 1))
                    print(f"Added edge between {genre1} and {genre2} with distance {distance}")

    return G

# Scrape away
genre_graph = scrape_genre_relationships_main_page(genre_dict)
print(f"Generated genre graph with {len(genre_graph.nodes)} genres and {len(genre_graph.edges)} relationships")

# Train node2vec on the genre graph
node2vec = Node2Vec(genre_graph, dimensions=16, walk_length=5, num_walks=20, workers=4) # tude for better embeddings
model = node2vec.fit(window=5, min_count=1)
genre_embeddings = {genre: model.wv[genre] for genre in genre_graph.nodes}

# Save embeddings to cache (NOT NEEDED ANYMORE)
with open("genre_embeddings.pkl", "wb") as f:
    pickle.dump(genre_embeddings, f)

# Save genre embeddings to json
with open("genre_embeddings.json", "w") as f:
    json.dump({k: v.tolist() for k, v in genre_embeddings.items()}, f)

print("Genre embeddings saved to genre_embeddings.json")