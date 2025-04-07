import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys
from io import StringIO

import app as heaterfy_engine

color_sequence = ['#ff0000','#ff5a00','#ff9a00','#ffce00','#ffe808','#FF6347','#FF7F50','#FFA500','#FF8C00','#FF0000']

st.set_page_config(
    page_title="Heaterfy - Playlist Recommender",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Displaying first five recommendations
# TODO: Remove???
if 'rec_display_count' not in st.session_state:
    st.session_state.rec_display_count = 5
if 'cosine_display_count' not in st.session_state:
    st.session_state.cosine_display_count = 5
if 'ridge_display_count' not in st.session_state:
    st.session_state.ridge_display_count = 5
if 'nn_display_count' not in st.session_state:
    st.session_state.nn_display_count = 5
if 'glm_display_count' not in st.session_state:
    st.session_state.glm_display_count = 5
if 'cluster_expanded' not in st.session_state:
    st.session_state.cluster_expanded = {}


st.markdown("""
<style>
/* Spotify embed width */
iframe {
    max-width: 1000px !important;
    width: 100% !important;
}

.block-container {
    max-width: 1200px !important;
}

[data-testid="column"] {
    width: 100% !important;
    max-width: 100% !important;
}

.element-container {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# For errors
class CaptureOutput:
    def __init__(self):
        self.output = StringIO()
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.output.write(text)
        self.stdout.write(text)
        
    def flush(self):
        self.stdout.flush()
        
    def release(self):
        sys.stdout = self.stdout
        return self.output.getvalue()

@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "heaterfy_catalogue_final2.csv")
    return pd.read_csv(dataset_path)

def get_recommendations(playlist_df, threshold=0.60, recency_bonus=0.0, progress_callback=None):
    capture = CaptureOutput()
    
    try:
        results = {
            "logs": [],
            "figures": {},
            "top_tracks": None,
            "all_recommendations": None,
            "feature_summary": None,
            "cluster_summary": None,
            "clustering_data": None
        }
    
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, "datasets", "heaterfy_catalogue_final2.csv")
        
        # Create a temporary file to save the playlist
        tmp_playlist_path = os.path.join(base_dir, "tmp_playlist.csv")
        playlist_df.to_csv(tmp_playlist_path, index=False)
        
        # Run recommendation engine
        try:
            result = heaterfy_engine.main(dataset_path, tmp_playlist_path, threshold, recency_bonus=recency_bonus)            
            clustering_results = None
            
            if isinstance(result, tuple):
                if len(result) == 3:
                    filtered_songs, updated_playlist_df, clustering_results = result
                    playlist_df = updated_playlist_df
                elif len(result) == 2:
                    filtered_songs, updated_playlist_df = result
                    playlist_df = updated_playlist_df
                    clustering_results = None
                else:
                    filtered_songs = result[0]
                    clustering_results = None
            else:
                filtered_songs = result
                clustering_results = None
            
            # Check if recommendations are valid
            if filtered_songs is None or filtered_songs.empty:
                results["logs"].append("No recommendations were generated")
                st.error("No recommendations were generated")
                return results
                
        except Exception as e:
            import traceback
            error_msg = f"Error in recommendation engine: {str(e)}\n{traceback.format_exc()}"
            results["logs"].append(error_msg)
            st.error(f"Error: {str(e)}")
            return results
        
        # Store clustering results
        if clustering_results:
            results["clustering_data"] = clustering_results
        
        # Extract  results
        results["all_recommendations"] = filtered_songs
        results["top_tracks"] = filtered_songs.head(20).copy()
        results["updated_playlist"] = playlist_df
        
        audio_features = [
            "Danceability", "Energy", "Tempo", "Valence", "Liveness", 
            "Acousticness", "Instrumentalness", "Speechiness", "Loudness"
        ]
        
        available_features = [f for f in audio_features if f in playlist_df.columns]
        if available_features:
            results["feature_summary"] = playlist_df[available_features].mean()
        
        if 'cluster' in filtered_songs.columns and 'cluster' in playlist_df.columns:
            playlist_cluster_dist = playlist_df['cluster'].value_counts(normalize=True).to_dict()
            results["cluster_summary"] = {
                "original_dist": playlist_cluster_dist,
                "cluster_counts": playlist_df['cluster'].value_counts().to_dict()
            }
            
            # Create cluster visualization
            if len(available_features) >= 2:
                pca = PCA(n_components=2)
                valid_data = filtered_songs.dropna(subset=available_features)
                
                if len(valid_data) > 10:
                    pca_result = pca.fit_transform(valid_data[available_features])
                    
                    # PCA results
                    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                    pca_df['cluster'] = valid_data['cluster'].values
                    pca_df['Track Name'] = valid_data['Track Name'].values
                    pca_df['Artist Name(s)'] = valid_data['Artist Name(s)'].values
                    
                    if 'combined_score' in valid_data.columns:
                        pca_df['score'] = valid_data['combined_score'].values
                    else:
                        pca_df['score'] = 50  # Default
                    
                    # Plot ploty scatter plot plot
                    fig_clusters = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2', 
                        color='cluster',
                        color_discrete_sequence=color_sequence,
                        size='score',
                        hover_data=['Track Name', 'Artist Name(s)', 'score'],
                        title='Track Clusters (PCA)',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
                    )
                    results["figures"]["clusters"] = fig_clusters
        
        # Year distribution
        if 'Release Year' in filtered_songs.columns:
            fig_years = px.histogram(
                filtered_songs, 
                x='Release Year',
                color_discrete_sequence=color_sequence,
                nbins=30,
                title="📆 Release Year Distribution",
                labels={"Release Year": "Year", "count": "Number of Tracks"}
            )
            
            if 'Release Year' in playlist_df.columns:
                avg_year = playlist_df['Release Year'].mean()
                fig_years.add_vline(x=avg_year, line_dash="dash", line_color="red", 
                                   annotation_text=f"Playlist Avg: {avg_year:.0f}")
            
            results["figures"]["year_distribution"] = fig_years
        
        # Audio features radar chart
        if available_features:
            fig_radar = go.Figure()
            
            # Add playlist average
            fig_radar.add_trace(go.Scatterpolar(
                r=playlist_df[available_features].mean().values,
                theta=available_features,
                fill='toself',
                name='Playlist Average'
            ))
            
            # Top 5 recommendations
            if len(filtered_songs) >= 5:
                top5 = filtered_songs.head(5)
                for i, (idx, row) in enumerate(top5.iterrows()):
                    feature_values = []
                    for feat in available_features:
                        if feat in row and not pd.isna(row[feat]):
                            feature_values.append(row[feat])
                        else:
                            feature_values.append(0)
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=feature_values,
                        theta=available_features,
                        name=f"Top Rec #{i+1}",
                        opacity=0.7
                    ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                color_discrete_sequence=color_sequence,
                title="Audio Features Comparison"
            )
            results["figures"]["audio_features"] = fig_radar
        
        # Genre info
        if 'Genres' in playlist_df.columns:
            try:
                genre_counts = playlist_df['Genres'].dropna().str.split(',').explode().str.strip().value_counts()
                if len(genre_counts) > 0:
                    results["genre_summary"] = genre_counts.head(10)
            except Exception as e:
                print(f"Error extracting genres: {str(e)}")
        
        # Clean up temporary file
        if os.path.exists(tmp_playlist_path):
            os.remove(tmp_playlist_path)
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating recommendations: {str(e)}\n{traceback.format_exc()}"
        results["logs"].append(error_msg)
    
    # Clean again
    logs = capture.release()
    results["logs"] = logs.split('\n')
    
    return results

def styled_header(text):
    st.markdown(f"<h1 style='font-family:Metropolis,Gotham,sans-serif;'>{text}</h1>", unsafe_allow_html=True)

def styled_subheader(text):
    st.markdown(f"<h2 style='font-family:Metropolis,Gotham,sans-serif;'>{text}</h2>", unsafe_allow_html=True)

def main():
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    # Load CSS
    with open(os.path.join(os.path.dirname(__file__), "assets", "style.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Montserrat', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "upload_attempted" not in st.session_state:
        st.session_state.upload_attempted = False
    
    # Create sidebar
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "heaterfy_logo.png")
        st.image(logo_path, width=300, use_container_width=True) # logo
        
        st.markdown("""
        <div style="background-color: #212124; padding: 2px; border-radius: 2px; margin-bottom: 25px; text-align: center; font-size: 64px;">
        <h1 style="color: #FFFFFF;">Heaterfy</h1>
        </div>
        """, unsafe_allow_html=True)
         # #D91E41
        # st.header("Heaterfy")
        
        # Navigation menu
        page = st.radio(
            "",
            ["🎵 Playlist Recommender", "⚙️ How It Works", "🧼 Catalogue Cleaner", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        # Settings section
        st.header("Settings")
        threshold = st.slider(
            "🧬 Similarity Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.6, 
            step=0.05, 
            help="Initial filtering via XGBoost classification model. Removes tracks below this threshold."
        )
        
        # Model weights tuning
        st.markdown("### Model Weights ⚖️")
        cosine_weight = st.slider(
            "📐 Cosine Similarity Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.05, 
            help="Unsupervised model which compares audio features only"
        )
        fitness_weight = st.slider(
            "⛰️ Ridge Score Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05, 
            help="Regression model based on audio features only"
        )
        nn_weight = st.slider(
            "🧠 Neural Network Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.05, 
            help="Neural network with combined scores from other models"
        )
        glm_weight = st.slider(
            "📈 GLM Score Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.05, 
            step=0.05, 
            help="General Linear Model based on metadata"
        )
        
        # Normalize weights to ensure they sum to 1
        total_weight = cosine_weight + fitness_weight + nn_weight + glm_weight
        if total_weight > 0:
            cosine_weight /= total_weight
            fitness_weight /= total_weight
            nn_weight /= total_weight
            glm_weight /= total_weight
            
        st.markdown("### Time Preferences ⏰")
        recency_bonus = st.slider(
            "🕰️ Recency Bonus", 
            min_value=-1.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.1, 
            help="Positive values favor newer tracks, negative values favor older, 0 is neutral"
        )
        
        # Display time preference
        if recency_bonus > 0:
            st.markdown(f"🆕 **Favoring newer bangers** (+{recency_bonus:.1f})")
        elif recency_bonus < 0:
            st.markdown(f"🏺 **Favoring older classics** ({recency_bonus:.1f})")
        else:
            st.markdown("🤷‍♂️ **Meh. No time preference**")
        
        st.markdown("---")
        st.markdown("### Final Weights")
        st.write(f"Cosine Similarity: {cosine_weight:.2f}")
        st.write(f"Ridge Score: {fitness_weight:.2f}")
        st.write(f"Neural Network: {nn_weight:.2f}")
        st.write(f"GLM Score: {glm_weight:.2f}")
    
    # Playlist Recommender
    if page == "🎵 Playlist Recommender":
        styled_header("Heaterfy")
        st.markdown("<p style='font-size:26px;'>🎵 Playlist Recommender</p>", unsafe_allow_html=True)
        st.markdown("Upload your Spotify playlist and get song recommendations! 🔥🔥🔥")
        
        uploaded_file = st.file_uploader("Insert CSV here", type="csv")
        
        if uploaded_file is not None:
            # Load the uploaded playlist
            try:
                playlist_df = pd.read_csv(uploaded_file)
                
                # Display playlist info
                st.subheader("Your Playlist")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tracks", len(playlist_df))
                
                with col2:
                    if 'Genres' in playlist_df.columns:
                        unique_genres = set()
                        for genres in playlist_df['Genres'].dropna():
                            if isinstance(genres, str):
                                unique_genres.update([g.strip() for g in genres.split(',')])
                        st.metric("Unique Genres", len(unique_genres))
                
                # Display playlist preview
                with st.expander("Playlist Preview", expanded=False):
                    st.dataframe(playlist_df)
                    
                    st.subheader("🎧 Preview Tracks on Spotify")
                    for i, (_, row) in enumerate(playlist_df.head(5).iterrows()):
                        if 'Track ID' in row and pd.notna(row['Track ID']):
                            track_url = f"https://open.spotify.com/embed/track/{row['Track ID']}"
                            st.markdown(f"**{i+1}. {row['Track Name']} - {row['Artist Name(s)']}**")
                            st.components.v1.iframe(track_url, height=80)  # Embed Spotify player
                
                # Generate heat button
                generate_clicked = st.button("Generate Heat", key="gen_rec_btn")

                if generate_clicked or st.session_state.current_results is not None:
                    if generate_clicked:
                        progress_text = st.empty()
                        progress_text.text("Starting recommendation engine...")
                        
                        # Run the engine
                        with st.spinner("Analyzing your playlist and generating recommendations..."):
                            start_time = time.time()
                            results = get_recommendations(playlist_df, threshold, recency_bonus=recency_bonus)
                            end_time = time.time()
                        
                        # Store results in session state
                        st.session_state.current_results = results
                        
                        progress_text.empty()
                        st.success(f"Recommendations generated in {end_time - start_time:.2f} seconds!")
                    else:
                        # Use cached results from session state
                        results = st.session_state.current_results
                    
                    # Create tabs for individual model results
                    tabs = st.tabs(["Recommendations", "Playlist Insights", "Advanced Analysis", "Model Performance", "Logs"])
                    
                    # TAB 1: Recommendations
                    with tabs[0]:
                        if results["top_tracks"] is not None and not results["top_tracks"].empty:
                            st.subheader("🏆 Top 20 Recommendations")
                            
                            # Display top tracks with relevant columns
                            display_cols = ['Track Name', 'Artist Name(s)']
                            if 'Release Year' in results["top_tracks"].columns:
                                display_cols.append('Release Year')
                            
                            score_cols = [col for col in results["top_tracks"].columns if 'score' in col.lower()]
                            display_cols.extend(score_cols)
                            display_df = results["top_tracks"][display_cols].copy()
                            
                            # Format score columns
                            for col in score_cols:
                                if display_df[col].dtype == 'float64':
                                    display_df[col] = display_df[col].round(1)
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Spotify links
                            if 'Track ID' in results["top_tracks"].columns:
                                st.subheader("🎧 Top Recommendations")
                                
                                # Display top 10 with Spotify embeds
                                for i, (_, row) in enumerate(results["top_tracks"].head(10).iterrows()):
                                    if pd.notna(row['Track ID']):
                                        track_url = f"https://open.spotify.com/embed/track/{row['Track ID']}"
                                        st.markdown(f"**{i+1}. {row['Track Name']} - {row['Artist Name(s)']}**")
                                        st.components.v1.iframe(track_url, height=80)
                                
                                # Display next 10 as links only
                                st.subheader("More Recommendations")
                                if len(results["top_tracks"]) > 10:
                                    for i, (_, row) in enumerate(results["top_tracks"].iloc[10:20].iterrows(), start=11):
                                        if pd.notna(row['Track ID']):
                                            track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                            st.markdown(f"**{i}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})**")
                            
                            # Download option
                            if results["all_recommendations"] is not None:
                                csv = results["all_recommendations"].to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download All Recommendations",
                                    csv,
                                    "heaterfy_recommendations.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                        else:
                            st.error("No recommendations were generated. Check the logs for details.")
                            
                    # TAB 2: Playlist Insights
                    with tabs[1]:
                        st.subheader("Playlist Insights")
                        
                        st.subheader("🎨 Genre Distribution")
                        try:
                            # Extract genres directly from the input playlist
                            if 'Genres' in playlist_df.columns:
                                genre_counts = playlist_df['Genres'].dropna().str.split(',').explode().str.strip().value_counts().head(10)
                                
                                if len(genre_counts) > 0:
                                    # Create the chart
                                    genres = list(genre_counts.index)
                                    counts = list(genre_counts.values)
                                    
                                    # horizontal bar chart
                                    fig_genres = go.Figure()
                                    fig_genres.add_trace(go.Bar(
                                        x=counts,
                                        y=genres,
                                        orientation='h',
                                        marker=dict(
                                            color=color_sequence[0],
                                            line=dict(color='rgb(248, 248, 249)', width=1)
                                        )
                                    ))
                                    
                                    fig_genres.update_layout(
                                        title="\t    Top Genres in Your Playlist",
                                        xaxis_title="Number of Tracks",
                                        yaxis_title="",
                                        plot_bgcolor="#212124",
                                        paper_bgcolor="#212124",
                                        font=dict(color="white"),
                                        height=400,
                                        margin=dict(l=10, r=10, t=30, b=10)
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig_genres, use_container_width=True)
                                else:
                                    st.info("No genre data found in your playlist. All genres are either missing or empty.")
                            else:
                                st.info("Your playlist doesn't have a Genres column :(")
                                
                        except Exception as e:
                            st.error(f"Error analyzing genres: {str(e)}")
                            with st.expander("View error details"):
                                st.code(f"{type(e).__name__}: {str(e)}")
                            
                        # Audio feature summary
                        if results["feature_summary"] is not None:
                            st.subheader("🎵 Audio Features")
                            feature_df = pd.DataFrame(results["feature_summary"]).reset_index()
                            feature_df.columns = ['Feature', 'Value']
                            
                            # Add a column to label the values
                            def interpret_feature(row):
                                feature = row['Feature']
                                value = row['Value']
                                
                                if feature == "Loudness":
                                    return "High" if value > -8 else "Low" if value < -12 else "Neutral"
                                elif feature == "Tempo":
                                    return "Fast" if value > 120 else "Slow" if value < 90 else "Moderate"
                                else:
                                    return "High" if value > 0.6 else "Low" if value < 0.4 else "Neutral"
                                
                            feature_df['Level'] = feature_df.apply(interpret_feature, axis=1)
                            display_feature_df = feature_df[~feature_df['Feature'].isin(['Tempo', 'Loudness'])]
                            
                            # Create feature display
                            col1, col2 = st.columns(2)

                            with col1:                            
                                # Display average Tempo and Loudness
                                tempo_val = feature_df.loc[feature_df['Feature'] == 'Tempo', 'Value'].values
                                loudness_val = feature_df.loc[feature_df['Feature'] == 'Loudness', 'Value'].values
                                if len(tempo_val) > 0 and len(loudness_val) > 0:
                                    # Tempo
                                    st.markdown(f"""
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <h3 style="font-size: 24px; color: white; margin: 0;">Mean Tempo</h3>
                                        <p style="font-size: 32px; font-weight: bold; color: #FFA500; margin: 0;">{tempo_val[0]:.1f} BPM</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Loudness
                                    st.markdown(f"""
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <h3 style="font-size: 24px; color: white; margin: 0;">Mean Loudness</h3>
                                        <p style="font-size: 32px; font-weight: bold; color: #FFA500; margin: 0;">{loudness_val[0]:.1f} dB</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with col2:
                                # Radar chart
                                feature_radar = go.Figure()
                                radar_features = feature_df[~feature_df['Feature'].isin(['Tempo', 'Loudness'])]
                                
                                feature_radar.add_trace(go.Scatterpolar(
                                    r=radar_features['Value'].values,
                                    theta=radar_features['Feature'].values,
                                    fill='toself',
                                    name='Your Playlist'
                                ))
                                
                                feature_radar.update_layout(
                                    polar=dict(radialaxis=dict(visible=True)),
                                    showlegend=False,
                                    title="\t    Audio Features Profile",
                                    plot_bgcolor="#161618",
                                    paper_bgcolor="#161618",
                                )
                                
                                st.plotly_chart(feature_radar, use_container_width=True, key="radar_chart")
                                st.dataframe(display_feature_df, use_container_width=True)
                    
                    # TAB 3: Advanced Analysis
                    with tabs[2]:
                        st.subheader("Advanced Analysis")
                        
                        # Year distribution
                        if "figures" in results and "year_distribution" in results["figures"]:
                            st.subheader("📆 Release Year Distribution")
                            fig_years = results["figures"]["year_distribution"]
                            fig_years.update_traces(marker=dict(color=color_sequence[1]))
                            fig_years.update_layout(
                                plot_bgcolor="#30122d",
                                paper_bgcolor="#30122d",
                                font=dict(color="white")
                            )
                            st.plotly_chart(fig_years, use_container_width=True, key="year_dist")
                        
                        # Cluster Analysis
                        if results["cluster_summary"] is not None:
                            st.subheader("🌌 Cluster Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                # Create pie chart or donut? for clusters
                                cluster_data = results["cluster_summary"]["cluster_counts"]
                                cluster_df = pd.DataFrame(list(cluster_data.items()), columns=['Cluster', 'Count'])

                                fig_cluster_pie = px.pie(
                                    cluster_df,
                                    values='Count',
                                    names='Cluster',
                                    title='\t    Playlist Diversity (Clusters)',
                                    hole=0.4,
                                    color_discrete_sequence=color_sequence
                                )
                                fig_cluster_pie.update_traces(
                                    textinfo='percent+label',
                                    textfont_color='white',
                                    insidetextfont=dict(color='white'),
                                    outsidetextfont=dict(color='black'),
                                )
                                fig_cluster_pie.update_layout(
                                    plot_bgcolor="#212124",
                                    paper_bgcolor="#212124",
                                    font=dict(color="white")
                                )
                                    
                                st.plotly_chart(fig_cluster_pie, use_container_width=True, key="cluster_pie")
                            
                            with col2:
                                updated_playlist = results.get("updated_playlist", playlist_df)
        
                                if 'cluster' in updated_playlist.columns:
                                    features_for_viz = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness']
                                    available_features = [f for f in features_for_viz if f in updated_playlist.columns]
                                    
                                    if len(available_features) >= 2:
                                        pca = PCA(n_components=2)
                                        valid_data = updated_playlist.dropna(subset=available_features)
                                        
                                        if len(valid_data) > 5:
                                            pca_result = pca.fit_transform(valid_data[available_features])
                                            
                                            # Generate results
                                            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                                            pca_df['cluster'] = valid_data['cluster'].values
                                            pca_df['Track Name'] = valid_data['Track Name'].values
                                            pca_df['Artist Name(s)'] = valid_data['Artist Name(s)'].values
                                            
                                            # Visualization
                                            fig_clusters = px.scatter(
                                                pca_df,
                                                x='PC1',
                                                y='PC2',
                                                color='cluster',
                                                size=[10] * len(pca_df),
                                                hover_data=['Track Name', 'Artist Name(s)'],
                                                title='\t    Your Playlist Clusters',
                                                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                                                color_discrete_sequence=color_sequence
                                            )
                                            
                                            fig_clusters.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))
                                            fig_clusters.update_layout(
                                                legend_title_text='Cluster',
                                                plot_bgcolor="#161618",
                                                paper_bgcolor="#161618",
                                                font=dict(color="white")
                                            )
                                            
                                            st.plotly_chart(fig_clusters, use_container_width=True, key="pca_clusters")
                                        else:
                                            st.info("Not enough complete data points for cluster visualization")
                                    else:
                                        st.info("Not enough audio features to visualize clusters")
                                else:
                                    st.info("Cluster information not available in your playlist. This is expected when using small playlists or when clustering fails.")
                            
                            # Elbow graph
                            st.subheader("🔭 Optimal Cluster Detection")
                            if "clustering_data" in results and results["clustering_data"] and "wcss" in results["clustering_data"]:
                                k_range = list(range(1, len(results["clustering_data"]["wcss"]) + 1))
                                wcss = results["clustering_data"]["wcss"]
                                optimal_k = results["clustering_data"]["optimal_k"] if "optimal_k" in results["clustering_data"] else None
                            else:
                                # just show something
                                k_range = list(range(1, 11))
                                wcss = [100, 80, 60, 45, 40, 38, 36, 35, 34, 33]
                                optimal_k = 4  # default clusyters
                            
                            if wcss and len(wcss) > 0:
                                fig_elbow = px.line(
                                    x=k_range, 
                                    y=wcss,
                                    markers=True,
                                    color_discrete_sequence=color_sequence,
                                    title='\t    Elbow Method',
                                    labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'}
                                )
                                fig_elbow.update_layout(
                                    legend_title_text='Elbow',
                                    plot_bgcolor="#161618",
                                    paper_bgcolor="#161618",
                                    font=dict(color="white")
                                )
                                
                                # Add a vertical line
                                if optimal_k:
                                    fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="yellow", 
                                                    annotation_text=f"Optimal k={optimal_k}")
                                
                                st.plotly_chart(fig_elbow, use_container_width=True, key="elbow_chart")
                                
                                # Write explanation about clusters for tunes
                                st.info("Clusters represent different 'moods' or 'styles' in your playlist. " +
                                        "A diverse playlist has multiple clusters, while a focused playlist may have fewer. " +
                                        "The elbow method helps determine the optimal number of clusters.")
                            else:
                                st.info("Elbow curve data not available. This would show how the optimal number of clusters was determined.")
                        
                        # Other visualizations?
                        if results["figures"]:
                            for name, fig in results["figures"].items():
                                if name not in ["audio_features", "year_distribution", "clusters"]:
                                    st.subheader(name.replace("_", " ").title())
                                    st.plotly_chart(fig, use_container_width=True, key=f"other_viz_{name}")
                    
                    # TAB 4: Model Performance
                    with tabs[3]:
                        st.subheader("Model Performance")
                        model_tabs = st.tabs(["Cosine Similarity", "Ridge Model", "Neural Network", "GLM", "Clusters"])
                        
                        # Check if we have score columns
                        score_cols = {}
                        if results["top_tracks"] is not None and not results["top_tracks"].empty:
                            for col in results["top_tracks"].columns:
                                if 'cosine' in col.lower():
                                    score_cols['cosine'] = col
                                elif 'ridge' in col.lower() or 'fitness' in col.lower():
                                    score_cols['ridge'] = col
                                elif 'neural' in col.lower() or 'nn' in col.lower():
                                    score_cols['neural'] = col
                                elif 'glm' in col.lower():
                                    score_cols['glm'] = col
                        
                        # Cosine Similarity Tab
                        with model_tabs[0]:
                            st.subheader("Top Tracks by Cosine Similarity")
                            if 'cosine' in score_cols and results["all_recommendations"] is not None:
                                # Sort by cosine score
                                cosine_tracks = results["all_recommendations"].sort_values(by=score_cols['cosine'], ascending=False).head(20)
                                
                                # Display tracks and hyperlinks
                                for i, (_, row) in enumerate(cosine_tracks.iterrows()):
                                    if 'Track ID' in row and pd.notna(row['Track ID']):
                                        track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                        score = f"{row[score_cols['cosine']]:.2f}"
                                        st.markdown(f"**{i+1}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})** (Score: {score})")
                            else:
                                st.info("Cosine similarity scores not available in recommendations")
                        
                        # Ridge Model Tab
                        with model_tabs[1]:
                            st.subheader("Top Tracks by Ridge Model")
                            if 'ridge' in score_cols and results["all_recommendations"] is not None:
                                # Sort by ridge score
                                ridge_tracks = results["all_recommendations"].sort_values(by=score_cols['ridge'], ascending=False).head(20)
                                
                                # Display tracks and hyperlinks
                                for i, (_, row) in enumerate(ridge_tracks.iterrows()):
                                    if 'Track ID' in row and pd.notna(row['Track ID']):
                                        track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                        score = f"{row[score_cols['ridge']]:.2f}"
                                        st.markdown(f"**{i+1}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})** (Score: {score})")
                            else:
                                st.info("Ridge model scores not available in recommendations")
                        
                        # Neural Network Tab
                        with model_tabs[2]:
                            st.subheader("Top Tracks by Neural Network")
                            if 'neural' in score_cols and results["all_recommendations"] is not None:
                                # Sort by neural network score
                                nn_tracks = results["all_recommendations"].sort_values(by=score_cols['neural'], ascending=False).head(20)
                                
                                # Display tracks and hyperlinks
                                for i, (_, row) in enumerate(nn_tracks.iterrows()):
                                    if 'Track ID' in row and pd.notna(row['Track ID']):
                                        track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                        score = f"{row[score_cols['neural']]:.2f}"
                                        st.markdown(f"**{i+1}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})** (Score: {score})")
                            else:
                                st.info("Neural network scores not available in recommendations")

                        # GLM Tab
                        with model_tabs[3]:
                            st.subheader("Top Tracks by GLM")
                            if 'glm' in score_cols and results["all_recommendations"] is not None:
                                # Sort by glm score
                                glm_tracks = results["all_recommendations"].sort_values(by=score_cols['glm'], ascending=False).head(20)
                                
                                # Display tracks and hyperlinks
                                for i, (_, row) in enumerate(glm_tracks.iterrows()):
                                    if 'Track ID' in row and pd.notna(row['Track ID']):
                                        track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                        score = f"{row[score_cols['glm']]:.2f}"
                                        st.markdown(f"**{i+1}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})** (Score: {score})")
                            else:
                                st.info("GLM scores not available in recommendations")
                        
                        # Clusters Tab
                        with model_tabs[4]:
                            st.subheader("Representative Tracks from Each Cluster")
                            if 'cluster' in results["all_recommendations"].columns:
                                # Get all clusters
                                clusters = results["all_recommendations"]['cluster'].unique()
                                
                                # For each cluster, show tracks with hyperlink
                                for cluster in sorted(clusters):
                                    with st.expander(f"Cluster {cluster}", expanded=False):
                                        cluster_tracks = results["all_recommendations"][results["all_recommendations"]['cluster'] == cluster]
                                        if 'combined_score' in cluster_tracks.columns:
                                            cluster_tracks = cluster_tracks.sort_values(by='combined_score', ascending=False)
                                        
                                        # Display number of tracks in this cluster
                                        st.write(f"Total tracks in cluster: {len(cluster_tracks)}")
                                        
                                        # Show top 5 tracks with hyperlink
                                        for i, (_, row) in enumerate(cluster_tracks.head(5).iterrows()):
                                            if 'Track ID' in row and pd.notna(row['Track ID']):
                                                track_link = f"https://open.spotify.com/track/{row['Track ID']}"
                                                score_text = f"{row['combined_score']:.2f}" if 'combined_score' in row else "N/A"
                                                st.markdown(f"**{i+1}. [{row['Track Name']} - {row['Artist Name(s)']}]({track_link})** (Score: {score_text})")
                            else:
                                st.info("Cluster information is not available in the recommendations")
                                
                    # Logs tab
                    with tabs[4]:
                        st.subheader("Process Logs")
                        
                        log_container = st.container()
                        with log_container:
                            for log in results["logs"]:
                                st.text(log)
            
            except Exception as e:
                st.error(f"Error processing the playlist: {str(e)}")
                st.info("Make sure your CSV has the correct format with at least 'Track Name' and 'Artist Name(s)' columns.")
        else:
            if st.session_state.upload_attempted:
                st.info("No playlist uploaded. Please upload a CSV file to get recommendations.")
                
                with st.expander("CSV Format Requirements"):
                    st.markdown("""
                    get them from spotify lol
                    """)
            else:
                # welcoming first-time message
                st.markdown("""
                <div style="background-color: #212124; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <h2 style="color: #FFA500;">Welcome to Heaterfy! 🔥</h2>
                    <p style="font-size: 18px; margin-bottom: 15px;">Upload your Spotify playlist CSV to discover new tracks. Its time we get our $hit together.</p>
                    <p style="font-size: 16px;">Use the file uploader above to get started.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("CSV Format Requirements"):
                st.markdown("""
                get them from spotify lol
                """)
    elif page == "⚙️ How It Works":
        styled_header("How It Works")
        
        st.markdown("""
        ## 🔥 The Heaterfy Recommendation Engine
        
        Our recommendation system uses a multi-model system that feeds off of each other to provide the best recommendations possible:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Initial Filtering
            - 🧬 **Similarity Filter** - Initial XGBoost classification
            - 🎯 **Threshold Application** - Removes dissimilar tracks
            """)
        
        with col2:
            st.markdown("""
            ### Advanced Models
            - 📐 **Cosine Similarity** - Compares audio features with playlist centroid
            - ⛰️ **Ridge Regression** - Predicts audio feature fit via L2 regression
            - 📈 **GLM Model** - Analyzes metadata patterns
            - 🌌 **Clustering** - Identifies diversity in the playlist #inclusive #ally
            - 🧠 **Neural Network** - Makes final reccomendation based on previous scores
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ## Technical Details
        
        ### Audio Features Used
        - **Tempo** - Beats per minute
        - **Energy** - Intensity and activity
        - **Danceability** - How suitable for dancing
        - **Valence** - Musical positiveness/brightness
        - **Acousticness** - Acoustic vs. electric sonically
        - **Instrumentalness** - Lack of vocal presence
        - **Speechiness** - Spoken word presence
        """)

    elif page == "ℹ️ About":
        styled_header("About Heaterfy")
        
        st.markdown("""
        <div style="background-color: #212124; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h2 style="color: #FFA500;">Our Mission</h2>
        <p style="font-size: 18px;">Sick of streaming services recommending TRASH 🗑️ ?</p>
        <p style="font-size: 18px;">We gotchyu fam. 👌</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Features")
            st.markdown("""
            - **Multi-model recommendation engine**
            - **Customizable model weights**
            - **Time preference controls**
            - **Cluster-based track diversity**
            - **Detailed audio feature analysis**
            - **Full Spotify integration**
            """)
    elif page == "🧼 Catalogue Cleaner":
        st.header("🧼 Catalogue Cleaner")
        dirty_df_file = st.file_uploader("Insert disgusting CSV here", type="csv")
        
        if dirty_df_file is not None:
            # Load the uploaded playlist
            try:
                dirty_df = pd.read_csv(dirty_df_file)
                
                # genre_column = "Artist Genres"
                # artist_column = "Artist Name(s)"

                # # Check if the columns exist in the DataFrame
                # if genre_column not in dirty_df.columns or artist_column not in dirty_df.columns:
                #     raise KeyError(f"Columns '{genre_column}' or '{artist_column}' not found in the DataFrame")

                # # Convert genre column from CSV string to list
                # def parse_genre(genre_str):
                #     return genre_str.split(',') if isinstance(genre_str, str) else []

                # dirty_df[genre_column] = dirty_df[genre_column].apply(parse_genre)

                # # Extract all unique genres
                # all_genres = set()
                # for genres in dirty_df[genre_column]:
                #     all_genres.update(genres)

                # all_genres = sorted(all_genres)

                # # Extract all unique artists
                # all_artists = sorted(dirty_df[artist_column].dropna().unique())

                # st.title("Filter Songs by Genre and Artist")
                # # Multi-select checkboxes for genres
                # selected_genres = st.multiselect("Select genres to remove:", all_genres)

                # # Multi-select checkboxes for artists
                # selected_artists = st.multiselect("Select artists to remove:", all_artists)

                # # Apply genre filter
                # if selected_genres:
                #     dirty_df = dirty_df[~dirty_df[genre_column].apply(lambda genre_list: any(genre in genre_list for genre in selected_genres))]

                # # Apply artist filter
                # if selected_artists:
                #     dirty_df = dirty_df[~dirty_df[artist_column].isin(selected_artists)]

                # # Show filtered dataframe
                # st.write("Filtered Data:", dirty_df)

                # # Download option
                # st.download_button(
                #     label="Download Filtered Data as CSV",
                #     data=dirty_df.to_csv(index=False).encode("utf-8"),
                #     file_name=f"{dirty_df_file}_filtered_songs.csv",
                #     mime="text/csv"
                # )
            except:
                st.error("Error loading the CSV file. Please check the file and try again.")
                return
            

if __name__ == "__main__":
    main()