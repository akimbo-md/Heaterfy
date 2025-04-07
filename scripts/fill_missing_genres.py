import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import os
import requests
import time
import sys
import numpy as np
from difflib import SequenceMatcher
import random
from requests.exceptions import Timeout
import logging
from datetime import datetime, timedelta
import re
from difflib import get_close_matches

# 🔑 Set up API credentials
SPOTIPY_CLIENT_ID = "626c10e1b6f841bc86cecd7e95d14180"
SPOTIPY_CLIENT_SECRET = "f47360ffa3ca47cebfcb1e75c67a0126"
REDIRECT_URI = "http://localhost:8888/callback"
LASTFM_API_KEY = "41b5cd1b9c42e223f2528c369e1ef05b"
DISCOGS_TOKEN = "ebgAXLtTgLvdFbLaZtQvLLUTUCJwftRNZkRoKuYQ"

# Add user agent to avoid 403 errors
headers = {
    'User-Agent': 'HeaterFy/1.0 +http://example.com'
}

# Initialize Spotify API with backoff disabled to handle rate limits ourselves
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, 
    client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager,
    retries=0,  # Disable automatic retries
    requests_timeout=3.0  # Set a 3-second timeout
)

# 📂 Load dataset
csv_filename = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\emotional_dance.csv"
progress_filename = "C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\emotional_dance_updated.csv"

# Load the dataset
df = pd.read_csv(csv_filename)

# Load progress if it exists
if os.path.exists(progress_filename):
    progress_df = pd.read_csv(progress_filename)
    df.update(progress_df)
    print(f"🔄 Loaded progress from {progress_filename}")

# Track API rate limits
LASTFM_DISABLED_UNTIL = None
DISCOGS_DISABLED_UNTIL = 8000
SPOTIFY_DISABLED_UNTIL = None  # Add this new variable
MUSICBRAINZ_DISABLED_UNTIL = None
API_COOLDOWN = 60  # Default cooldown in seconds when we hit rate limits

def save_progress(df, filename):
    """Saves the current progress to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"\n💾 Progress saved to {filename}")

def countdown_timer(seconds):
    """Displays a countdown timer for the specified number of seconds."""
    for remaining in range(seconds, 0, -1):
        print(f"\r⏳ Waiting for {remaining} seconds...", end="")
        time.sleep(1)
    print("\r⏳ Waiting complete. Resuming...")

def handle_rate_limit(response):
    """Handles API rate limit by checking response headers and waiting if necessary."""
    if response.status_code == 429:  # 429 Too Many Requests
        retry_after = int(response.headers.get("Retry-After", 60))  # Default to 60 seconds if not specified
        print(f"🔄 Rate limit reached. Waiting for {retry_after} seconds before retrying...")
        countdown_timer(retry_after)
        return True
    return False

def get_spotify_genres(track_id, primary_artist_name):
    """Fetches genres from Spotify using Track ID with timeout handling."""
    global SPOTIFY_DISABLED_UNTIL
    
    # Check if Spotify is currently rate-limited
    if SPOTIFY_DISABLED_UNTIL:
        current_time = datetime.now()
        if current_time < SPOTIFY_DISABLED_UNTIL:
            wait_time = (SPOTIFY_DISABLED_UNTIL - current_time).total_seconds()
            print(f"⏱️ Spotify is rate limited. Will be available in {wait_time:.0f} seconds.")
            return None
        else:
            # Reset the disabled flag
            SPOTIFY_DISABLED_UNTIL = None
    
    try:
        # 🔍 Search for the track on Spotify
        track_info = sp.track(track_id)
        album_id = track_info["album"]["id"]
        artists = track_info["artists"]  # All contributing artists

        # 🎵 Fetch Album Genres
        album_info = sp.album(album_id)
        album_genres = album_info["genres"]

        # 🎤 Fetch All Artist Genres
        all_artist_genres = []
        for artist in artists:
            artist_info = sp.artist(artist["id"])
            all_artist_genres.extend(artist_info["genres"])

        # Track Genre → Album Genre → Artist Genres
        if album_genres:
            print(f"✅ Found Album Genres for {track_id}: {album_genres}")
            return ", ".join(album_genres)
        elif all_artist_genres:
            print(f"✅ Found Artist Genres for {track_id}: {list(set(all_artist_genres))}")
            return ", ".join(set(all_artist_genres))  # Remove duplicates
        else:
            print(f"⚠️ No Spotify genres found for {track_id}")
            return None

    except spotipy.exceptions.SpotifyException as e:
        # Handle rate limit errors
        if e.http_status == 429:
            retry_after = int(e.headers.get('Retry-After', 60)) if hasattr(e, 'headers') else 60
            # Cap very long waits to 10 minutes
            if retry_after > 600:
                retry_after = 600
            
            SPOTIFY_DISABLED_UNTIL = datetime.now() + timedelta(seconds=retry_after)
            print(f"⚠️ Spotify rate limited. Disabling for {retry_after} seconds.")
            return None
        else:
            print(f"❌ Spotify API error for {track_id}: {e}")
            return None
    except requests.exceptions.Timeout:
        print(f"⏱️ Spotify request timed out for {track_id}")
        return None
    except Exception as e:
        print(f"❌ Error fetching Spotify data for {track_id}: {e}")
        return None

def get_lastfm_genres(artist_name):
    """Fetches genres from Last.fm with a strict timeout and better error handling."""
    global LASTFM_DISABLED_UNTIL
    
    # Check if Last.fm is currently rate-limited
    if LASTFM_DISABLED_UNTIL:
        current_time = datetime.now()
        if current_time < LASTFM_DISABLED_UNTIL:
            wait_time = (LASTFM_DISABLED_UNTIL - current_time).total_seconds()
            print(f"⏱️ Last.fm is rate limited. Will be available in {wait_time:.0f} seconds.")
            return None
        else:
            # Reset the disabled flag
            LASTFM_DISABLED_UNTIL = None
    
    try:
        # Escape special characters in artist name for URL
        escaped_artist = requests.utils.quote(artist_name)
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist={escaped_artist}&api_key={LASTFM_API_KEY}&format=json"
        
        # Use a very strict timeout (1.5 seconds)
        response = requests.get(url, timeout=1.5, headers=headers)
        
        # Immediately check for rate limiting response codes
        if response.status_code == 429:
            # Disable Last.fm for a while
            LASTFM_DISABLED_UNTIL = datetime.now() + timedelta(minutes=10)
            print(f"⚠️ Last.fm rate limited (status 429). Disabling for 10 minutes.")
            return None
            
        # Check for other error responses
        if response.status_code != 200:
            print(f"⚠️ Last.fm returned status {response.status_code}. Skipping.")
            return None
            
        # Process successful response - wrap JSON parsing in try/except
        try:
            data = response.json()
        except ValueError:
            print(f"⚠️ Invalid JSON response from Last.fm")
            return None
            
        # Check for error messages in the response body
        if "error" in data:
            error_code = data.get("error")
            error_msg = data.get("message", "Unknown error")
            print(f"⚠️ Last.fm API error {error_code}: {error_msg}")
            
            # Handle rate limiting error
            if error_code == 29:  # Rate limit error
                LASTFM_DISABLED_UNTIL = datetime.now() + timedelta(minutes=15)
                print("⚠️ Last.fm rate limited (error 29), disabling for 15 minutes")
                
            # If we see any log message about rate limits, disable for a while
            elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
                LASTFM_DISABLED_UNTIL = datetime.now() + timedelta(minutes=30)
                print(f"⚠️ Last.fm rate limit message detected: '{error_msg}'. Disabling for 30 minutes.")
                
            return None
        
        # Extract genre tags if available
        if "artist" in data and "tags" in data["artist"]:
            genres = [tag["name"].lower() for tag in data["artist"]["tags"]["tag"]]
            
            # Ensure the genre list is not empty before returning
            if genres:
                print(f"✅ Found Last.fm genres for {artist_name}: {genres}")
                return ", ".join(genres)
            else:
                print(f"⚠️ Last.fm returned an empty genre list for {artist_name}")
                
        # Retry using the first artist name if multiple artists were in the field
        if "," in artist_name and not LASTFM_DISABLED_UNTIL:
            first_artist = artist_name.split(",")[0].strip()
            print(f"⚠️ No Last.fm genres found for {artist_name}. Retrying with primary artist: {first_artist}")
            return get_lastfm_genres(first_artist)
            
        print(f"⚠️ No valid genre data found on Last.fm for {artist_name}")
        return None

    except Timeout:
        print(f"⏱️ Last.fm request timed out for {artist_name}")
        # Implement a backoff strategy - disable for 5 minutes after timeouts
        LASTFM_DISABLED_UNTIL = datetime.now() + timedelta(minutes=5)
        print("⚠️ Too many timeouts - disabling Last.fm for 5 minutes")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching Last.fm genres for {artist_name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error fetching Last.fm genres for {artist_name}: {e}")
        return None

def get_discogs_genres(artist_name, track_name):
    """Fetches genres from Discogs API."""
    try:
        # Search for the release
        search_url = f"https://api.discogs.com/database/search?q={artist_name}+{track_name}&type=release&token={DISCOGS_TOKEN}"
        response = requests.get(search_url, timeout=3, headers=headers)
        
        if response.status_code != 200:
            print(f"⚠️ Discogs API returned status {response.status_code}")
            return None
            
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            # Get the first result's genres and styles
            result = data["results"][0]
            genres = result.get("genre", [])
            styles = result.get("style", [])
            
            # Combine genres and styles
            all_genres = genres + styles
            
            if all_genres:
                print(f"✅ Found Discogs genres for {artist_name} - {track_name}: {all_genres}")
                return ", ".join(all_genres)
        
        print(f"⚠️ No Discogs genres found for {artist_name} - {track_name}")
        return None
        
    except Timeout:
        print(f"⏱️ Discogs request timed out for {artist_name} - {track_name}")
        return None
    except Exception as e:
        print(f"❌ Error fetching Discogs genres for {artist_name} - {track_name}: {e}")
        return None

def get_musicbrainz_genres(artist_name, track_name):
    """Fetches genres from MusicBrainz API which has more lenient rate limits."""
    global MUSICBRAINZ_DISABLED_UNTIL
    
    # Check if MusicBrainz is currently rate-limited
    if MUSICBRAINZ_DISABLED_UNTIL:
        current_time = datetime.now()
        if current_time < MUSICBRAINZ_DISABLED_UNTIL:
            wait_time = (MUSICBRAINZ_DISABLED_UNTIL - current_time).total_seconds()
            print(f"⏱️ MusicBrainz is rate limited. Will be available in {wait_time:.0f} seconds.")
            return None
        else:
            MUSICBRAINZ_DISABLED_UNTIL = None
    
    try:
        # MusicBrainz requires a proper user-agent with contact info
        mb_headers = {
            'User-Agent': 'HeaterFy/1.0 (contact@example.com)',
            'Accept': 'application/json'
        }
        
        # First try: search for the specific recording (track)
        search_url = f"https://musicbrainz.org/ws/2/recording/?query={requests.utils.quote(track_name)}%20AND%20artist:{requests.utils.quote(artist_name)}&fmt=json"
        response = requests.get(search_url, headers=mb_headers, timeout=3)
        
        # Handle rate limiting
        if response.status_code == 429:
            # MusicBrainz typically asks for 1 request per second
            MUSICBRAINZ_DISABLED_UNTIL = datetime.now() + timedelta(seconds=5)
            print(f"⚠️ MusicBrainz rate limited. Disabling for 5 seconds.")
            return None
        
        # Handle other errors
        if response.status_code != 200:
            print(f"⚠️ MusicBrainz API error: {response.status_code}")
            return None
        
        data = response.json()
        
        # Check if we got any recordings
        if 'recordings' in data and len(data['recordings']) > 0:
            # Look for the best match
            best_match = None
            
            for recording in data['recordings']:
                # Check if artist names match
                recording_artists = [artist['name'].lower() for artist in recording.get('artist-credit', [])]
                
                # Split our artist into components (for collaborations)
                artist_parts = re.split(r'[,&]|\bfeat\.|\bfeaturing\b|\bwith\b|\bpresents\b', artist_name.lower())
                artist_parts = [part.strip() for part in artist_parts if part.strip()]
                
                # Check if at least one artist part matches
                matches = False
                for artist_part in artist_parts:
                    if any(artist_part in ra for ra in recording_artists):
                        matches = True
                        break
                
                if matches and recording.get('title', '').lower() == track_name.lower():
                    best_match = recording
                    break
            
            if not best_match and len(data['recordings']) > 0:
                best_match = data['recordings'][0]  # Fall back to first result
            
            if best_match:
                # Get the release group to find genres
                if 'releases' in best_match and len(best_match['releases']) > 0:
                    release_id = best_match['releases'][0]['id']
                    
                    # Get release info
                    release_url = f"https://musicbrainz.org/ws/2/release/{release_id}?inc=genres+release-groups+artists+artist-credits&fmt=json"
                    release_response = requests.get(release_url, headers=mb_headers, timeout=3)
                    
                    if release_response.status_code == 200:
                        release_data = release_response.json()
                        
                        # Try to get genres from different sources in order of specificity
                        found_genres = []
                        
                        # 1. Check release genres
                        if 'genres' in release_data and len(release_data['genres']) > 0:
                            found_genres.extend([genre['name'] for genre in release_data['genres']])
                        
                        # 2. Check release group genres
                        if 'release-group' in release_data and 'genres' in release_data['release-group']:
                            found_genres.extend([genre['name'] for genre in release_data['release-group']['genres']])
                        
                        # 3. Check artist genres
                        if 'artist-credit' in release_data and len(release_data['artist-credit']) > 0:
                            artist_id = release_data['artist-credit'][0]['artist']['id']
                            
                            # Get artist info with genres
                            artist_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=genres&fmt=json"
                            artist_response = requests.get(artist_url, headers=mb_headers, timeout=3)
                            
                            if artist_response.status_code == 200:
                                artist_data = artist_response.json()
                                if 'genres' in artist_data and len(artist_data['genres']) > 0:
                                    found_genres.extend([genre['name'] for genre in artist_data['genres']])
                        
                        # Remove duplicates and return
                        if found_genres:
                            unique_genres = list(set(found_genres))
                            print(f"✅ Found MusicBrainz genres for {track_name}: {unique_genres}")
                            return ", ".join(unique_genres)
        
        # Second try: If no track found, try artist search
        if "," in artist_name:
            primary_artist = artist_name.split(",")[0].strip()
        else:
            primary_artist = artist_name
            
        artist_url = f"https://musicbrainz.org/ws/2/artist/?query={requests.utils.quote(primary_artist)}&fmt=json"
        artist_response = requests.get(artist_url, headers=mb_headers, timeout=3)
        
        if artist_response.status_code == 200:
            artist_data = artist_response.json()
            if 'artists' in artist_data and len(artist_data['artists']) > 0:
                # Find the most relevant artist
                best_artist = None
                for artist in artist_data['artists']:
                    if artist['name'].lower() == primary_artist.lower():
                        best_artist = artist
                        break
                
                if not best_artist and len(artist_data['artists']) > 0:
                    best_artist = artist_data['artists'][0]  # Fall back to first result
                
                if best_artist:
                    # Get genres for this artist
                    artist_id = best_artist['id']
                    detailed_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=genres&fmt=json"
                    detailed_response = requests.get(detailed_url, headers=mb_headers, timeout=3)
                    
                    if detailed_response.status_code == 200:
                        detailed_data = detailed_response.json()
                        if 'genres' in detailed_data and len(detailed_data['genres']) > 0:
                            genres = [genre['name'] for genre in detailed_data['genres']]
                            print(f"✅ Found MusicBrainz genres for artist {primary_artist}: {genres}")
                            return ", ".join(genres)
        
        print(f"⚠️ No genres found on MusicBrainz for {track_name} by {artist_name}")
        return None
        
    except Timeout:
        print(f"⏱️ MusicBrainz request timed out for {artist_name} - {track_name}")
        return None
    except Exception as e:
        print(f"❌ Error fetching MusicBrainz data: {e}")
        return None

def validate_genre_match(track_name, artist_name, inferred_genres):
    """Validate if the inferred genres make sense for this track/artist."""
    # List of broad genre categories and their common subgenres
    genre_categories = {
        'electronic': ['edm', 'dance', 'house', 'techno', 'trance', 'dubstep', 'electro', 'electronic', 'electronica'],
        'rock': ['rock', 'metal', 'punk', 'alternative', 'indie', 'grunge', 'hardcore'],
        'hip_hop': ['hip hop', 'rap', 'trap', 'r&b', 'rnb', 'rhythm and blues'],
        'pop': ['pop', 'synth-pop', 'dance-pop', 'k-pop', 'j-pop'],
        'jazz': ['jazz', 'fusion', 'bebop', 'swing'],
        'classical': ['classical', 'orchestra', 'chamber', 'symphony', 'opera'],
        'folk': ['folk', 'acoustic', 'country', 'bluegrass', 'singer-songwriter']
    }
    
    # Check if the genres belong to conflicting categories
    genre_list = inferred_genres.lower().split(', ')
    
    # Count which categories these genres belong to
    category_counts = {}
    for genre in genre_list:
        for category, subgenres in genre_categories.items():
            if any(subgenre in genre for subgenre in subgenres):
                category_counts[category] = category_counts.get(category, 0) + 1
    
    # If we have clear domination of one category, it's probably correct
    if category_counts:
        top_category = max(category_counts.items(), key=lambda x: x[1])
        top_count = top_category[1]
        total_matches = sum(category_counts.values())
        
        # If more than 70% of genres match one category, it's probably right
        if top_count / total_matches > 0.7:
            return True
            
    # For dance music, check track name for dance-related words
    dance_terms = ['dance', 'club', 'remix', 'mix', 'beat', 'groove', 'dj', 'edm', 'party']
    if any(term in track_name.lower() for term in dance_terms):
        # If track name suggests dance but genres don't include it, be suspicious
        if 'electronic' not in category_counts:
            return False
    
    # The validation is inconclusive
    return None

def find_similar_tracks(track_name, artist_name, df):
    """Find tracks by the same artist and use their genres with validation."""
    # Get all tracks with non-empty genres
    tracks_with_genres = df[df["Genres"].notna() & (df["Genres"].str.strip() != "")].copy()
    
    if len(tracks_with_genres) == 0:
        return None
    
    # First try: Look for exact artist matches
    same_artist_tracks = tracks_with_genres[tracks_with_genres["Artist Name(s)"].str.lower() == artist_name.lower()]
    
    # If we found tracks by the same artist, use those
    if len(same_artist_tracks) > 0:
        print(f"✅ Found {len(same_artist_tracks)} other tracks by {artist_name}")
        
        # Extract all genres from the artist's tracks
        all_genres = []
        for _, row in same_artist_tracks.iterrows():
            genres = row["Genres"].split(", ")
            all_genres.extend(genres)
        
        # Count genre occurrences
        genre_counts = {}
        for genre in all_genres:
            genre = genre.lower().strip()
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Take the most common genres (max 3)
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_genres:
            genres_list = [genre for genre, _ in top_genres]
            inferred_genres = ", ".join(genres_list)
            print(f"📊 Inferred genres for {track_name} based on other tracks by {artist_name}: {genres_list}")
            
            # Validate the genre match
            is_valid = validate_genre_match(track_name, artist_name, inferred_genres)
            if is_valid is False:
                print(f"⚠️ Genre validation failed for {track_name}. These genres may be incorrect.")
                return None
                
            return inferred_genres
    
    # Second try: For artists with multiple names or featuring artists, try checking the first artist name
    if "," in artist_name or "&" in artist_name or "feat" in artist_name.lower():
        # Split and get the first artist
        if "," in artist_name:
            first_artist = artist_name.split(",")[0].strip()
        elif "&" in artist_name:
            first_artist = artist_name.split("&")[0].strip()
        elif "feat" in artist_name.lower():
            first_artist = artist_name.split("feat")[0].strip()
        else:
            first_artist = artist_name
            
        print(f"🔍 Trying with primary artist: {first_artist}")
        
        # Look for tracks by this artist - be more strict about match quality
        primary_artist_tracks = tracks_with_genres[
            tracks_with_genres["Artist Name(s)"].str.lower().str.contains(r'\b' + re.escape(first_artist.lower()) + r'\b')
        ]
        
        if len(primary_artist_tracks) > 0:
            print(f"✅ Found {len(primary_artist_tracks)} tracks by primary artist {first_artist}")
            
            # Extract all genres from the primary artist's tracks
            all_genres = []
            for _, row in primary_artist_tracks.iterrows():
                genres = row["Genres"].split(", ")
                all_genres.extend(genres)
            
            # Count genre occurrences
            genre_counts = {}
            for genre in all_genres:
                genre = genre.lower().strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Take the most common genres (max 3)
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_genres:
                genres_list = [genre for genre, _ in top_genres]
                inferred_genres = ", ".join(genres_list)
                print(f"📊 Inferred genres for {track_name} based on tracks by {first_artist}: {genres_list}")
                
                # Validate the genre match
                is_valid = validate_genre_match(track_name, artist_name, inferred_genres)
                if is_valid is False:
                    print(f"⚠️ Genre validation failed for {track_name}. These genres may be incorrect.")
                    return None
                    
                return inferred_genres
    
    print(f"⚠️ Could not find other tracks by the same artist ({artist_name})")
    return None

def get_genre_for_track(track_id, artist_name, track_name, idx):
    """Tries multiple sources to find genres with better prioritization of track-specific lookups."""
    
    # Track-specific lookups first (across all APIs)
    
    # 1. Try Spotify first for track-specific genres (if not rate limited)
    if not SPOTIFY_DISABLED_UNTIL:
        print(f"🔄 Trying Spotify for track {track_name}")
        genre = get_spotify_genres(track_id, artist_name)
        if genre:
            return genre
    else:
        print(f"⏭️ Skipping Spotify (rate limited)")
    
    # 2. Try Discogs for track-specific release
    if not DISCOGS_DISABLED_UNTIL:
        print(f"🔄 Trying Discogs for {track_name} by {artist_name}")
        genre = get_discogs_genres(artist_name, track_name)
        if genre:
            return genre
    else:
        print(f"⏭️ Skipping Discogs (rate limited)")
    
    # 3. Try MusicBrainz for track-specific lookup
    if not MUSICBRAINZ_DISABLED_UNTIL:
        print(f"🔄 Trying MusicBrainz for {track_name} by {artist_name}")
        genre = get_musicbrainz_genres(artist_name, track_name)
        if genre:
            return genre
    else:
        print(f"⏭️ Skipping MusicBrainz (rate limited)")
        
    # Add a small delay between API calls to avoid rate limiting
    time.sleep(0.5)
    
    # Now try artist-level lookups
    
    # 4. Try Last.fm for artist tags
    if not LASTFM_DISABLED_UNTIL:
        print(f"🔄 Trying Last.fm for artist {artist_name}")
        genre = get_lastfm_genres(artist_name)
        if genre:
            return genre
    else:
        print(f"⏭️ Skipping Last.fm (rate limited)")
    
    # Last resort: find similar tracks in our dataset by the same artist
    print(f"🔄 Looking for similar tracks by same artist to infer genres for {track_name}")
    genre = find_similar_tracks(track_name, artist_name, df)
    
    return genre if genre else "unknown"  # Return "unknown" if all methods fail

try:
    updated_count = 0
    for idx, row in df.iterrows():
        track_id = row["Track ID"]
        track_name = row["Track Name"]
        artist_name = row["Artist Name(s)"]
        current_genre = row["Genres"]
        
        if pd.isna(current_genre) or current_genre.strip() == "":
            print(f"\n🔄 Fetching genre for {track_name} by {artist_name} (Track ID: {track_id})")
            new_genre = get_genre_for_track(track_id, artist_name, track_name, idx)
            
            if new_genre and new_genre != "unknown":
                df.at[idx, "Genres"] = new_genre
                updated_count += 1
                print(f"✅ Updated genres for {track_name}: {new_genre}")

        # Save progress periodically
        if idx % 20 == 0:
            save_progress(df, progress_filename)
            
        # Random delay between 1-3 seconds to avoid hammering APIs
        time.sleep(random.uniform(1, 3))
        
except KeyboardInterrupt:
    print("\n\n🛑 STOP! Saving progress before exit...")
    save_progress(df, progress_filename)
    print("✅ Progress saved. Exiting.")
    sys.exit(0)

# Save the updated dataset
if updated_count > 0:
    save_progress(df, progress_filename)
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Saved updated dataset with {updated_count} filled genres: {csv_filename}")
else:
    print("\n✅ No new genres were added. Dataset remains unchanged.")