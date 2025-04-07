import requests
import time
import random
from datetime import datetime, timedelta

MUSICBRAINZ_DISABLED_UNTIL = None
headers = {
    'User-Agent': 'HeaterFy/1.0 (abmark88@gmail.com)',
    'Accept': 'application/json'
}

def get_musicbrainz_genres(artist_name, track_name):
    global MUSICBRAINZ_DISABLED_UNTIL
    if MUSICBRAINZ_DISABLED_UNTIL and datetime.now() < MUSICBRAINZ_DISABLED_UNTIL:
        return None
    
    MUSICBRAINZ_DISABLED_UNTIL = None
    
    try:
        search_url = f"https://musicbrainz.org/ws/2/recording/?query={requests.utils.quote(track_name)}%20AND%20artist:{requests.utils.quote(artist_name)}&fmt=json"
        response = requests.get(search_url, headers=headers, timeout=3)
        
        if response.status_code == 429:
            MUSICBRAINZ_DISABLED_UNTIL = datetime.now() + timedelta(seconds=5)
            return None
        if response.status_code != 200:
            return None
        
        data = response.json()
        if 'recordings' in data and data['recordings']:
            # Get first matching recording
            for recording in data['recordings']:
                if 'releases' in recording and recording['releases']:
                    release_id = recording['releases'][0]['id']
                    release_url = f"https://musicbrainz.org/ws/2/release/{release_id}?inc=genres+release-groups+artists&fmt=json"
                    
                    release_response = requests.get(release_url, headers=headers, timeout=3)
                    if release_response.status_code == 200:
                        release_data = release_response.json()
                        genres = []
                        
                        # Check release genres
                        if 'genres' in release_data:
                            genres.extend([g['name'] for g in release_data['genres']])
                        
                        # Check release group genres
                        if 'release-group' in release_data and 'genres' in release_data['release-group']:
                            genres.extend([g['name'] for g in release_data['release-group']['genres']])
                        
                        if genres:
                            return ", ".join(set(genres))
        
        # Try artist lookup if no track match
        primary_artist = artist_name.split(",")[0].strip() if "," in artist_name else artist_name
        artist_url = f"https://musicbrainz.org/ws/2/artist/?query={requests.utils.quote(primary_artist)}&fmt=json"
        artist_response = requests.get(artist_url, headers=headers, timeout=3)
        
        if artist_response.status_code == 200:
            artist_data = artist_response.json()
            if 'artists' in artist_data and artist_data['artists']:
                artist_id = artist_data['artists'][0]['id']
                genres_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=genres&fmt=json"
                
                genres_response = requests.get(genres_url, headers=headers, timeout=3)
                if genres_response.status_code == 200:
                    genres_data = genres_response.json()
                    if 'genres' in genres_data and genres_data['genres']:
                        return ", ".join([g['name'] for g in genres_data['genres']])
        
        return None
    except Exception:
        return None

def find_similar_artist_tracks(artist_name, df):
    tracks_with_genres = df[df["Genres"].notna() & (df["Genres"].str.strip() != "")]
    
    # Try exact artist match
    same_artist = tracks_with_genres[tracks_with_genres["Artist Name(s)"].str.lower() == artist_name.lower()]
    if len(same_artist) > 0:
        all_genres = []
        for _, row in same_artist.iterrows():
            genres = [g.strip() for g in row["Genres"].split(",")]
            all_genres.extend(genres)
        
        # Store genres the artist does
        genre_counts = {}
        for genre in all_genres:
            if genre:
                genre_counts[genre.lower()] = genre_counts.get(genre.lower(), 0) + 1
        
        # Get top genres
        if genre_counts:
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            return ", ".join(genre for genre, _ in top_genres)
    
    # Try with primary artist (sometimes there are more than one in the data)
    if "," in artist_name:
        primary_artist = artist_name.split(",")[0].strip()
        return find_similar_artist_tracks(primary_artist, df)
        
    return None

# Main function
def fill_missing_genres(playlist_df, catalog_df):
    print("Checking for missing genres...")
    
    if 'Genres' not in playlist_df.columns:
        playlist_df['Genres'] = ""
    
    # Find tracks with missing genres
    missing_mask = playlist_df['Genres'].isna() | (playlist_df['Genres'] == "")
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("All tracks have genre information")
        return playlist_df
    
    print(f"Found {missing_count} tracks with missing genres")
    
    # Process each missing track
    for idx in playlist_df[missing_mask].index:
        track_name = playlist_df.loc[idx, "Track Name"]
        artist_name = playlist_df.loc[idx, "Artist Name(s)"]
        
        print(f"Looking for genres: {track_name} by {artist_name}")
        genre = get_musicbrainz_genres(artist_name, track_name)
        if genre:
            print(f"Found MusicBrainz genres for {track_name}")
        else:
            # Try finding other tracks by same artist in catalog
            genre = find_similar_artist_tracks(artist_name, catalog_df)
            if genre:
                print(f"Found genres from other tracks by {artist_name}")
                
        # Update if we found genres
        if genre:
            playlist_df.loc[idx, "Genres"] = genre
            print(f"Updated genres: {genre}")
        else:
            print(f"Could not find genres for {track_name}")
            
        # api call limitting
        time.sleep(random.uniform(1.0, 2.0))
    
    # Summary
    filled_count = missing_count - (playlist_df['Genres'].isna() | (playlist_df['Genres'] == "")).sum()
    print(f"Successfully filled genres for {filled_count}/{missing_count} tracks")
    
    return playlist_df