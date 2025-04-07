import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import sparse

# Load data
print("Loading datasets...")
df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\trevor_ridge_scored_songs.csv")
playlist_df = pd.read_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\spotify_rips\\cmpt_455_-_trevors_heaters_embedded.csv")

# Add binary target for classification
playlist_df['is_playlist_song'] = 1  # Songs in playlist (positive examples)
df['is_playlist_song'] = 0  # Other songs (negative examples)

# Focus ONLY on metadata/categorical features and minimal audio features
# Different from the MLR model which likely emphasized audio characteristics
numerical_features = [
    "Duration (ms)", "Popularity"
]
categorical_features = ["Artist Name(s)", "Album Name", "Record Label", "Release Date"]

# Filter to columns that exist in both datasets
actual_numerical = [f for f in numerical_features if f in df.columns and f in playlist_df.columns]
actual_categorical = [f for f in categorical_features if f in df.columns and f in playlist_df.columns]

print(f"Using {len(actual_numerical)} numerical features: {actual_numerical}")
print(f"Using {len(actual_categorical)} categorical features: {actual_categorical}")

# Clean numerical features
for feature in actual_numerical:
    # Convert to numeric and handle missing values
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    playlist_df[feature] = pd.to_numeric(playlist_df[feature], errors='coerce')
    
    # Fill missing with median
    median = pd.concat([df[feature], playlist_df[feature]]).median()
    df[feature] = df[feature].fillna(median)
    playlist_df[feature] = playlist_df[feature].fillna(median)

# Clean categorical features
for feature in actual_categorical:
    # Convert all to string
    df[feature] = df[feature].fillna("Unknown").astype(str)
    playlist_df[feature] = playlist_df[feature].fillna("Unknown").astype(str)
    
    # Get values actually in the playlist
    playlist_values = set(playlist_df[feature].unique())
    
    # Mark values either as themselves (if in playlist) or as "Other"
    df[feature] = df[feature].apply(lambda x: x if x in playlist_values else "Other")
    
# Standardize numerical features
scaler = StandardScaler()
if actual_numerical:
    # Fit scaler on combined data
    all_numerical = pd.concat([df[actual_numerical], playlist_df[actual_numerical]])
    scaler.fit(all_numerical)
    
    # Transform both datasets
    df[actual_numerical] = scaler.transform(df[actual_numerical])
    playlist_df[actual_numerical] = scaler.transform(playlist_df[actual_numerical])

# For better balance, upsample the playlist data more aggressively
upsampled_playlist = pd.concat([playlist_df] * 5)  # Repeat playlist 5 times (more weight on playlist)
print(f"Original playlist size: {len(playlist_df)}, Upsampled: {len(upsampled_playlist)}")

# Prepare combined dataset for one-hot encoding
combined_df = pd.concat([df, upsampled_playlist], ignore_index=True)
feature_matrices = []

# Add numerical features
if actual_numerical:
    feature_matrices.append(combined_df[actual_numerical])

# Create additional metadata match features
for feature in actual_categorical:
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=2)
    encoded = encoder.fit_transform(combined_df[[feature]])
    feature_names = [f"{feature}_{val}" for val in encoder.categories_[0]]
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=combined_df.index)
    feature_matrices.append(encoded_df)
    
    # Also add a binary feature for "is in playlist"
    playlist_values = set(playlist_df[feature].unique())
    combined_df[f'{feature}_in_playlist'] = combined_df[feature].apply(lambda x: 1 if x in playlist_values else 0)
    feature_matrices.append(combined_df[[f'{feature}_in_playlist']])

# Combine all feature matrices
X_combined = pd.concat(feature_matrices, axis=1)

# Add constant term for intercept
X_combined = sm.add_constant(X_combined)

# Split back into training datasets
X = X_combined 
y = combined_df['is_playlist_song']

print(f"Final feature matrix shape: {X.shape}")

# Train GLM model with very light regularization to allow categorical features to have influence
print("Training GLM model...")
try:
    # Use very light regularization 
    glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(
        alpha=0.01,  # Very light regularization to allow categorical features to have effect
        L1_wt=0,     # Pure L2 regularization 
        maxiter=300  # More iterations for convergence
    )
    
    # Check for non-zero coefficients
    non_zero_coefs = [(name, coef) for name, coef in zip(X.columns, glm_model.params) if abs(coef) > 1e-3]
    print(f"\nFound {len(non_zero_coefs)} non-zero coefficients out of {len(glm_model.params)}")
    
    # Print top positive coefficients
    print("\nTop positive coefficients (features that increase playlist fit):")
    for name, coef in sorted([(n, c) for n, c in non_zero_coefs if c > 0], key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {name}: {coef:.4f}")
        
    print("\nTop negative coefficients (features that decrease playlist fit):")
    for name, coef in sorted([(n, c) for n, c in non_zero_coefs if c < 0], key=lambda x: x[1])[:10]:
        print(f"  {name}: {coef:.4f}")
    
    # Prepare prediction data (original dataset only)
    X_pred = X_combined.iloc[:len(df)]
    
    # Generate predictions
    print("\nGenerating predictions...")
    df['fit_probability_glm'] = glm_model.predict(X_pred)
    df['fitness_score_glm'] = df['fit_probability_glm'] * 100  # Pure GLM score, no blending
    
except Exception as e:
    print(f"Error in GLM model: {e}")
    print("Falling back to metadata matching approach...")
    
    # Use a more sophisticated metadata matching approach
    print("Using weighted metadata matching...")
    
    # Define importance weights for each feature (can be tuned)
    feature_weights = {
        "Artist Name(s)": 0.5,
        "Album Name": 0.2,
        "Record Label": 0.2,
        "Release Date": 0.1
    }
    
    match_scores = []
    for _, row in df.iterrows():
        weighted_score = 0
        total_weight = 0
        
        for feature in actual_categorical:
            if feature in feature_weights:
                weight = feature_weights[feature]
                total_weight += weight
                
                # Check if this value appears in the playlist
                if row[feature] in playlist_df[feature].values:
                    weighted_score += weight
        
        # Normalize by total weight
        if total_weight > 0:
            match_scores.append(weighted_score / total_weight)
        else:
            match_scores.append(0)
    
    # Set the GLM score to be purely the metadata match score
    df['fitness_score_glm'] = np.array(match_scores) * 100

# Ensure we have a good range of scores (10-100)
min_score = df['fitness_score_glm'].min()
max_score = df['fitness_score_glm'].max()

if max_score - min_score < 30:
    print(f"Score range too narrow ({min_score:.2f}-{max_score:.2f}), rescaling...")
    df['fitness_score_glm'] = 10 + ((df['fitness_score_glm'] - min_score) / 
                                   max(0.001, max_score - min_score) * 90)

# Save the results
df.to_csv("C:\\Users\\AKIMBO-MSI\\Documents\\School\\Winter 2025\\CMPT 455\\Final Project\\Heaterfy\\datasets\\trevor_glm_filtered_dataset.csv", index=False)

# Print results
print("\nGLM Model Results:")
print(f"Number of songs analyzed: {len(df)}")
print(f"Average GLM fitness score: {df['fitness_score_glm'].mean():.2f}")
print(f"Score range: {df['fitness_score_glm'].min():.2f} - {df['fitness_score_glm'].max():.2f}")

# Distribution of scores
bins = [0, 20, 40, 60, 80, 100]
score_dist = pd.cut(df['fitness_score_glm'], bins=bins).value_counts().sort_index()
print("\nScore distribution:")
print(score_dist)

# Print top/bottom songs
print("\nTop 10 songs by GLM score:")
top_songs = df.sort_values('fitness_score_glm', ascending=False).head(10)[['Track Name', 'Artist Name(s)', 'fitness_score_glm']]
print(top_songs)

print("\nBottom 5 songs by GLM score:")
bottom_songs = df.sort_values('fitness_score_glm', ascending=True).head(5)[['Track Name', 'Artist Name(s)', 'fitness_score_glm']]
print(bottom_songs)