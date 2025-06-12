from flask import Flask, render_template, request, jsonify, Response, session, send_file
from flask_session import Session
from queue import Queue, Empty
import json
import traceback
import tempfile
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import os
import sys
import numpy as np
import pandas as pd
import umap
import openai
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan
import plotly.graph_objects as go
import pickle
import requests
from datetime import datetime, timedelta
import re

# Determine if running in Docker or local environment
IS_DOCKER = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true' or '/' in os.getcwd()
print(f"Running in {'Docker container' if IS_DOCKER else 'local environment'}")
print(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)

# Create and configure progress queue as part of app config
app.config['PROGRESS_QUEUE'] = Queue()

# Set base directories based on environment
if IS_DOCKER:
    base_dir = "/home/user/app"
    session_dir = os.path.join(base_dir, 'flask_session')
    data_dir = os.path.join(base_dir, 'data', 'visualizations')
else:
    base_dir = os.getcwd()
    session_dir = os.path.normpath(os.path.join(base_dir, 'flask_session'))
    data_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))

# Create required directories
os.makedirs(session_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Set stricter session configuration 
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=session_dir,
    SESSION_PERMANENT=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_USE_SIGNER=True,
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', os.urandom(24)),
    SESSION_REFRESH_EACH_REQUEST=True,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)
)
session_dir = os.path.normpath(os.path.join(base_dir, 'flask_session'))
data_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))

# Ensure directories exist with proper permissions
for directory in [session_dir, data_dir]:
    directory = os.path.normpath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Initialize session extension before any route handlers
Session(app)

@app.before_request 
def before_request():
    """Ensure session and viz_file path are properly initialized"""
    # Initialize permanent session
    session.permanent = True
    
    # Create new session ID if needed
    if not session.get('id'):
        session['id'] = os.urandom(16).hex()
        print(f"Created new session ID: {session['id']}")
        
        # Check for existing visualizations that can be used for this new session
        if IS_DOCKER:
            # Use Linux-style paths for Docker
            base_dir = "/home/user/app"
            data_dir = os.path.join(base_dir, 'data', 'visualizations')
            temp_dir = '/tmp'
        else:
            # Use platform-independent paths for local development
            base_dir = os.getcwd()
            data_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))
            temp_dir = tempfile.gettempdir()
        
        # Find the most recent visualization file
        most_recent_file = None
        most_recent_time = 0
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.startswith("patent_viz_") and filename.endswith(".json"):
                    file_path = os.path.join(data_dir, filename)
                    file_time = os.path.getmtime(file_path)
                    if file_time > most_recent_time:
                        most_recent_time = file_time
                        most_recent_file = file_path
        
        # Also check temp directory
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                if filename.startswith("patent_viz_") and filename.endswith(".json"):
                    file_path = os.path.join(temp_dir, filename)
                    file_time = os.path.getmtime(file_path)
                    if file_time > most_recent_time:
                        most_recent_time = file_time
                        most_recent_file = file_path
        
        if most_recent_file:
            print(f"Found existing visualization for new session: {most_recent_file}")
            # Copy this visualization for the new session
            try:
                # Create paths for the new session
                new_data_path = os.path.join(data_dir, f'patent_viz_{session["id"]}.json')
                new_temp_path = os.path.join(temp_dir, f'patent_viz_{session["id"]}.json')
                
                # Ensure directories exist
                os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
                os.makedirs(os.path.dirname(new_temp_path), exist_ok=True)
                
                # Read existing visualization
                with open(most_recent_file, 'r') as src:
                    viz_data = json.load(src)
                
                # Write to both locations for the new session
                with open(new_data_path, 'w') as f:
                    json.dump(viz_data, f)
                with open(new_temp_path, 'w') as f:
                    json.dump(viz_data, f)
                
                print(f"Copied existing visualization to new session files: {new_data_path} and {new_temp_path}")
            except Exception as e:
                print(f"Error copying existing visualization for new session: {e}")
    
    session_id = session['id']
    
    # Use the global IS_DOCKER variable that includes the '/' in os.getcwd() check
    print(f"Running in Docker environment: {IS_DOCKER}")
    
    # Set data directory paths based on environment
    if IS_DOCKER:
        # Use Linux-style paths for Docker
        base_dir = "/home/user/app"
        data_dir = os.path.join(base_dir, 'data', 'visualizations')
        temp_dir = '/tmp'
    else:
        # Use platform-independent paths for local development
        base_dir = os.getcwd()
        data_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))
        temp_dir = tempfile.gettempdir()
    
    # Create data directory if it doesn't exist
    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created/verified data directory: {data_dir}")
        # Debug directory contents
        print(f"Contents of data directory: {os.listdir(data_dir)}")
    except Exception as e:
        print(f"Error creating data directory: {e}")
    
    # Create file paths based on environment
    data_path = os.path.join(data_dir, f'patent_viz_{session_id}.json')
    temp_path = os.path.join(temp_dir, f'patent_viz_{session_id}.json')
    
    # Use normpath for Windows but not for Docker
    if not IS_DOCKER:
        data_path = os.path.normpath(data_path)
        temp_path = os.path.normpath(temp_path)
    
    print(f"Data path set to: {data_path}")
    print(f"Temp path set to: {temp_path}")
    
    # Check if visualization exists before updating paths
    data_exists = os.path.exists(data_path)
    temp_exists = os.path.exists(temp_path)
    print(f"Data file exists: {data_exists}")
    print(f"Temp file exists: {temp_exists}")
    
    if data_exists:
        print(f"Found visualization in data dir: {data_path}")
        # Ensure temp copy exists
        try:
            if not temp_exists:
                # Ensure temp directory exists
                temp_parent = os.path.dirname(temp_path)
                if not os.path.exists(temp_parent):
                    os.makedirs(temp_parent, exist_ok=True)
                    
                with open(data_path, 'r') as src:
                    with open(temp_path, 'w') as dst:
                        dst.write(src.read())
                print(f"Created temp backup: {temp_path}")
        except Exception as e:
            print(f"Warning: Failed to create temp backup: {e}")
    elif temp_exists:
        print(f"Found visualization in temp dir: {temp_path}")
        # Restore from temp
        try:
            with open(temp_path, 'r') as src:
                with open(data_path, 'w') as dst:
                    dst.write(src.read())
            print(f"Restored from temp to: {data_path}")
        except Exception as e:
            print(f"Warning: Failed to restore from temp: {e}")
    
    # Update session paths
    session['viz_file'] = data_path
    session['temp_viz_file'] = temp_path
    session.modified = True
    
    print(f"Session paths - Data: {data_path} (exists={os.path.exists(data_path)})")
    print(f"Session paths - Temp: {temp_path} (exists={os.path.exists(temp_path)})")

@app.after_request
def after_request(response):
    """Ensure session is saved after each request"""
    try:
        session.modified = True
        print(f"Session after request: {dict(session)}")
    except Exception as e:
        print(f"Error saving session: {e}")
    return response

# Get API keys from environment variables
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_PATENTS = 3000  # Maximum patents to process
MIN_PATENTS_FOR_GAPS = 3000  # Minimum patents needed for reliable gap detection
CACHE_FILE = 'patent_embeddings_cache.pkl'

if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

def load_cache():
    """Load cached embeddings from file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return {}

def save_cache(cache):
    """Save embeddings cache to file"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_embedding(text, cache):
    """Get embedding for text, using cache if available"""
    if not text or text.strip() == "":
        return None
        
    if text in cache:
        return cache[text]
    
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response['data'][0]['embedding']
        if embedding:  # Only cache if we got a valid embedding
            cache[text] = embedding
            save_cache(cache)  # Save cache after each new embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def search_patents(keywords, page_size=100):
    """
    Search patents using Google Patents and generate embeddings
    """
    # Load existing cache
    embedding_cache = load_cache()
    
    all_patents = []
    page = 1
    total_processed = 0
    
    while len(all_patents) < MAX_PATENTS:
        update_progress('search', 'processing', f'Fetching page {page} of patents...')
        
        # SerpApi Google Patents API endpoint
        api_url = "https://serpapi.com/search"
        
        params = {
            "engine": "google_patents",
            "q": keywords,
            "api_key": SERPAPI_API_KEY,
            "num": page_size,
            "start": (page - 1) * page_size
        }

        try:
            response = requests.get(api_url, params=params)
            response_data = response.json()
            
            if "error" in response_data:
                print(f"API returned error: {response_data['error']}")
                break
                
            patents_data = response_data.get('organic_results', [])
            
            if not patents_data:
                print(f"No more patents found on page {page}")
                break
                
            for idx, patent in enumerate(patents_data):
                if len(all_patents) >= MAX_PATENTS:
                    break
                    
                # Format filing date
                filing_date = patent.get('filing_date', '')
                filing_year = 'N/A'
                if filing_date:
                    try:
                        filing_year = datetime.strptime(filing_date, '%Y-%m-%d').year
                    except ValueError:
                        pass

                # Get assignee
                assignee = patent.get('assignee', ['N/A'])[0] if isinstance(patent.get('assignee'), list) else patent.get('assignee', 'N/A')

                # Format title and abstract for embedding
                title = patent.get('title', '').strip()
                abstract = patent.get('snippet', '').strip()  # SerpAPI uses 'snippet' for abstract
                combined_text = f"{title}\n{abstract}".strip()

                # Get embedding for combined text
                total_processed += 1
                if total_processed % 10 == 0:  # Update progress every 10 patents
                    update_progress('embedding', 'processing', f'Processing patent {total_processed} of {MAX_PATENTS}...')
                
                embedding = get_embedding(combined_text, embedding_cache)

                formatted_patent = {
                    'title': title,
                    'assignee': assignee,
                    'filing_year': filing_year,
                    'abstract': abstract,
                    'link': patent.get('patent_link', '') or patent.get('link', ''),  # SerpAPI provides patent_link or link
                    'embedding': embedding
                }
                all_patents.append(formatted_patent)
            
            print(f"Retrieved {len(patents_data)} patents from page {page}")
            
            # Check if there are more pages
            has_more = len(patents_data) >= page_size
            if not has_more:
                break
                
            page += 1
            
        except Exception as e:
            print(f"Error searching patents: {e}")
            break
    
    # Save final cache state
    save_cache(embedding_cache)
    
    print(f"Total patents retrieved and embedded: {len(all_patents)}")
    return all_patents

def analyze_patent_group(patents, group_type, label, max_retries=3):
    """Analyze patent groups using ChatGPT"""
    # Get titles and date range
    titles = "; ".join(patents['title'].tolist()[:3])
    years = f"{patents['year'].min()}-{patents['year'].max()}"
    
    prompts = {
        'cluster': (
            f"Patents: {titles}. Years: {years}\nSummarize in 2-3 sentences.",
            "Describe the key aspects."
        ),
        'transitional': (
            f"Patents: {titles}. Years: {years}\nSummarize in 2-3 sentences.",
            "Describe the key aspects."
        ),
        'innovation_subcluster': (
            f"Patents: {titles}. Years: {years}\nSummarize in 2-3 sentences.",
            "Describe the key aspects."
        )
    }
    
    base_prompt = prompts[group_type][0]
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompts[group_type][1]},
                    {"role": "user", "content": base_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0]['message']['content']
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2 ** (retry_count - 1))
            else:
                return "Analysis failed."

def create_3d_visualization(patents):
    """
    Create a 3D visualization of patent embeddings using UMAP and Plotly
    """
    # Initialize variables for tracking different point types
    df = pd.DataFrame(patents)
    df['point_type'] = 'cluster'  # Default type for all points
    transitional_areas = []  # Initialize empty list for transitional areas
    
    if not patents:
        return None
        
    update_progress('clustering', 'processing', 'Extracting embeddings...')
    
    # Extract embeddings and metadata
    embeddings = []
    metadata = []
    
    for patent in patents:
        if patent['embedding'] is not None:
            embeddings.append(patent['embedding'])
            abstract = patent['abstract']
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            
            metadata.append({
                'title': patent['title'],
                'assignee': patent['assignee'],
                'year': patent['filing_year'],
                'abstract': abstract,
                'link': patent['link']
            })
    
    if not embeddings:
        return None
        
    # Check if we have enough patents for reliable gap detection
    if len(embeddings) < MIN_PATENTS_FOR_GAPS:
        print(f"\nWarning: Dataset size ({len(embeddings)} patents) is below recommended minimum ({MIN_PATENTS_FOR_GAPS})")
        print("Underexplored area detection may be less reliable with smaller datasets")
        print("Consider:")
        print("1. Broadening your search terms")
        print("2. Including more patent categories")
        print("3. Expanding the time range")
        
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    update_progress('clustering', 'processing', 'Applying UMAP dimensionality reduction...')
    
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings_array)
    
    update_progress('clustering', 'processing', 'Performing DBSCAN clustering...')
    
    # Create DataFrame for plotting
    df = pd.DataFrame(metadata)
    df['x'] = embedding_3d[:, 0]
    df['y'] = embedding_3d[:, 1]
    df['z'] = embedding_3d[:, 2]
    
    # --- Improved HDBSCAN clustering logic for sparse region detection ---
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_3d)

    n_points = len(scaled_embeddings)
    update_progress('clustering', 'processing', f'Analyzing {n_points} patents for clustering...')
    
    # Dynamically set max_clusters and target_noise based on number of patents
    if n_points < 100:
        max_clusters = 4
        max_retries = 2
        target_noise_ratio = 0.08
    elif n_points < 500:
        max_clusters = 6
        max_retries = 3
        target_noise_ratio = 0.06
    elif n_points < 1000:
        max_clusters = 8
        max_retries = 4
        target_noise_ratio = 0.05
    else:
        max_clusters = 15  # Increased from 12 to force more granular clustering
        max_retries = 8   # More retries to find optimal clustering
        target_noise_ratio = 0.03  # Keep low noise ratio

    # Even more aggressive cluster parameters for large datasets
    if n_points >= 1000:
        min_cluster_size = max(5, int(n_points * 0.015))  # Further reduced to 1.5% for large datasets
        min_samples = max(3, int(min_cluster_size * 0.95))  # Increased to 0.95 for even stricter formation
    else:
        min_cluster_size = max(5, int(n_points * 0.02))  # 2% for smaller datasets
        min_samples = max(3, int(min_cluster_size * 0.9))  # 0.9 ratio for smaller datasets

    target_noise = int(n_points * target_noise_ratio)
    print(f"Initial HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}, max_clusters={max_clusters}, max_retries={max_retries}, target_noise={target_noise}")
    retry = 0
    clusters = None
    n_clusters = 0
    n_noise = 0
    best_result = None
    best_score = float('-inf')

    while retry < max_retries:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.03,  # Reduced further to force even tighter clusters
            cluster_selection_method='eom',
            metric='euclidean',
            prediction_data=True
        )
        clusters = hdb.fit_predict(scaled_embeddings)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        noise_ratio = n_noise / len(clusters)
        avg_cluster_size = (len(clusters) - n_noise) / n_clusters if n_clusters > 0 else float('inf')
        
        print(f"\nClustering Statistics (try {retry+1}):")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of patents in sparse regions: {n_noise}")
        print(f"Total number of patents: {len(clusters)}")
        print(f"Noise ratio: {noise_ratio:.2%}")
        print(f"Average cluster size: {avg_cluster_size:.1f} patents")
        
        update_progress('clustering', 'processing', 
            f'Optimizing clusters (attempt {retry + 1}/{max_retries}): ' +
            f'Found {n_clusters} clusters with avg size {avg_cluster_size:.1f} patents')
        
        # Calculate a score for this clustering result
        # Penalize both too many and too few clusters, and reward good noise ratio
        score = -abs(n_clusters - max_clusters) + \
                -abs(noise_ratio - target_noise_ratio) * 10 + \
                -abs(avg_cluster_size - (n_points / max_clusters)) / 10
        
        if score > best_score:
            best_score = score
            best_result = (clusters, n_clusters, n_noise, noise_ratio, avg_cluster_size)
        
        # Adjust parameters based on results
        if n_clusters > max_clusters:
            print("Too many clusters, increasing parameters more aggressively...")
            min_cluster_size = int(min_cluster_size * 1.5)  # More aggressive increase
            min_samples = int(min_samples * 1.4)
        elif n_clusters == 1 and avg_cluster_size > len(clusters) * 0.8:
            print("Single dominant cluster detected, adjusting for better separation...")
            min_cluster_size = max(5, int(min_cluster_size * 0.6))  # More aggressive decrease
            min_samples = max(3, int(min_samples * 0.6))
        elif n_noise < target_noise * 0.5:
            print("Too few noise points, adjusting parameters...")
            min_cluster_size = int(min_cluster_size * 1.2)
            min_samples = max(3, int(min_samples * 0.8))
        elif n_clusters < max_clusters * 0.5:
            print("Too few clusters, decreasing parameters...")
            min_cluster_size = max(5, int(min_cluster_size * 0.8))
            min_samples = max(3, int(min_samples * 0.7))
        else:
            print("Acceptable clustering found.")
            break
            
        retry += 1

    # Use the best result if we didn't find an acceptable one
    if retry == max_retries and best_result is not None:
        print("Using best clustering result found...")
        clusters, n_clusters, n_noise, noise_ratio, avg_cluster_size = best_result

    df['cluster'] = clusters

    # --- First gather all existing clusters and their sizes ---
    cluster_info = []
    for label in set(clusters):
        if label != -1:  # Skip noise points
            cluster_mask = clusters == label
            cluster_patents = df[cluster_mask]
            if len(cluster_patents) > 0:
                cluster_info.append((label, len(cluster_patents), cluster_patents))
    
    # Sort clusters by size in descending order
    cluster_info.sort(key=lambda x: x[1], reverse=True)
    
    print("\nCluster Size Distribution:")
    for i, (label, size, _) in enumerate(cluster_info):
        print(f"Cluster {i} (originally {label}): {size} patents")
    
    # Create mapping for new cluster IDs
    cluster_id_map = {old_label: i for i, (old_label, _, _) in enumerate(cluster_info)}
    
    # Update cluster IDs in DataFrame
    new_clusters = clusters.copy()
    for old_label, new_label in cluster_id_map.items():
        new_clusters[clusters == old_label] = new_label
    df['cluster'] = new_clusters
    
    update_progress('clustering', 'processing', 'Identifying technology clusters and underexplored areas...')
    
    # --- Initialize point types ---
    df['point_type'] = 'unassigned'  # Start with all points unassigned
    cluster_insights = []  # Initialize insights list
    
    # First handle clustered points
    total_clusters = len(cluster_info)
    for new_id, (_, size, cluster_patents) in enumerate(cluster_info):
        update_progress('clustering', 'processing', f'Analyzing cluster {new_id + 1} of {total_clusters} ({size} patents)...')
        description = analyze_patent_group(cluster_patents, 'cluster', new_id)
        df.loc[cluster_patents.index, 'point_type'] = 'cluster'  # Mark clustered points
        cluster_insights.append({
            'type': 'cluster',
            'id': int(new_id) + 1,  # Store as 1-based ID
            'size': size,
            'label': f"Cluster {new_id + 1}",
            'description': description
        })

    # --- Improved two-stage density analysis for noise points ---
    noise_mask = df['cluster'] == -1
    noise_points = scaled_embeddings[noise_mask]
    noise_indices = df[noise_mask].index
    dense_noise_indices = []  # Initialize empty list for dense noise points
    
    if len(noise_points) >= 3:
        update_progress('clustering', 'processing', f'Analyzing {len(noise_points)} potential underexplored areas...')
        print(f"\nStructural Analysis for Underexplored Area Detection:")
        
        # Initialize sparse indices
        true_sparse_indices = []
        
        # Stage 1: Calculate local and global density metrics
        n_neighbors = min(max(5, int(len(noise_points) * 0.05)), 15)
        print(f"Using {n_neighbors} nearest neighbors for density calculation")
        
        # Calculate local density for noise points
        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(noise_points)
        local_distances, local_indices = nbrs_local.kneighbors(noise_points)
        local_densities = 1 / (np.mean(local_distances, axis=1) + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Calculate distances to cluster centers and their densities
        cluster_centers = []
        cluster_densities = []  # Store density of each cluster
        for label in set(clusters) - {-1}:
            cluster_mask = clusters == label
            cluster_points = scaled_embeddings[cluster_mask]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
            
            # Calculate cluster density using its member points
            if len(cluster_points) > 1:
                nbrs_cluster = NearestNeighbors(n_neighbors=min(5, len(cluster_points))).fit(cluster_points)
                cluster_dists, _ = nbrs_cluster.kneighbors(cluster_points)
                cluster_density = 1 / (np.mean(cluster_dists) + 1e-6)
            else:
                cluster_density = 0
            cluster_densities.append(cluster_density)
        
        cluster_centers = np.array(cluster_centers)
        cluster_densities = np.array(cluster_densities)
        
        if len(cluster_centers) > 0:
            # Calculate distances and density ratios to nearest clusters
            nbrs_clusters = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(cluster_centers)
            cluster_distances, nearest_cluster_indices = nbrs_clusters.kneighbors(noise_points)
            cluster_distances = cluster_distances.flatten()
            
            # Get density of nearest cluster for each point
            nearest_cluster_densities = cluster_densities[nearest_cluster_indices.flatten()]
            
            # Calculate density ratios (local density / nearest cluster density)
            density_ratios = local_densities / (nearest_cluster_densities + 1e-6)
            
            print("\nDensity Analysis Statistics:")
            print(f"Mean local density: {np.mean(local_densities):.3f}")
            print(f"Mean cluster density: {np.mean(cluster_densities):.3f}")
            print(f"Mean density ratio: {np.mean(density_ratios):.3f}")
            
            # Identify structural gaps using multiple criteria with more sensitive thresholds
            # 1. Density Isolation: Points with very low density compared to clusters
            # 2. Spatial Isolation: Points far from both clusters and other noise points
            # 3. Structural Stability: Points whose local neighborhood is also sparse
            
            # Calculate isolation scores with more balanced thresholds
            density_isolation = density_ratios < np.percentile(density_ratios, 65)  # More balanced threshold
            spatial_isolation = cluster_distances > np.percentile(cluster_distances, 50)  # Median distance threshold
            
            # Calculate structural stability with more balanced criteria
            structural_stability = np.zeros(len(noise_points), dtype=bool)
            for i, neighbors in enumerate(local_indices):
                neighbor_densities = local_densities[neighbors]
                # Point is stable if its neighborhood is relatively sparse
                structural_stability[i] = np.mean(neighbor_densities) < np.percentile(local_densities, 50)  # Use median
            
            # Use more balanced criteria - only need to meet any 1 of 3 criteria initially
            candidate_sparse_indices = [
                idx for i, idx in enumerate(noise_indices)
                if sum([density_isolation[i], spatial_isolation[i], structural_stability[i]]) >= 1  # Only need 1 out of 3 criteria
            ]
            
            # Start by assuming all non-candidate points are dense noise
            dense_noise_indices = [idx for idx in noise_indices if idx not in candidate_sparse_indices]
            
            # Now calculate distances between candidates and dense noise points with more sensitive threshold
            min_distance_threshold = np.percentile(cluster_distances, 40)  # More sensitive threshold
            # Filter candidates based on distance from dense noise regions
            if len(candidate_sparse_indices) > 0 and len(dense_noise_indices) > 0:
                dense_noise_points = scaled_embeddings[dense_noise_indices]
                true_sparse_indices = []
                
                for idx in candidate_sparse_indices:
                    point = scaled_embeddings[idx].reshape(1, -1)
                    distances_to_dense = NearestNeighbors(n_neighbors=1).fit(dense_noise_points).kneighbors(point)[0][0]
                    if distances_to_dense > min_distance_threshold:
                        true_sparse_indices.append(idx)
                
                # Update dense_noise_indices to include rejected candidates
                rejected_indices = [idx for idx in candidate_sparse_indices if idx not in true_sparse_indices]
                dense_noise_indices.extend(rejected_indices)
            else:
                true_sparse_indices = candidate_sparse_indices
        else:
            # Fallback using only local density analysis
            density_threshold = np.percentile(local_densities, 25)  # Bottom 25% sparsest points
            true_sparse_indices = [idx for i, idx in enumerate(noise_indices) 
                                 if local_densities[i] < density_threshold]
            dense_noise_indices = [idx for idx in noise_indices if idx not in true_sparse_indices]
        
        print(f"\nFinal Classification:")
        print(f"True underexplored areas identified: {len(true_sparse_indices)}")
        print(f"Transitional areas identified: {len(dense_noise_indices)}")
        if len(true_sparse_indices) > 0:
            print(f"Underexplored area ratio: {len(true_sparse_indices)/len(noise_points):.2%}")
            print("\nUnderexplored Area Criteria Used:")
            print("1. Density Isolation: Significantly lower density than nearest cluster")
            print("2. Spatial Isolation: Far from both clusters and other points")
            print("3. Structural Stability: Forms stable sparse regions with neighbors")
        
        # Update point types in DataFrame for sparse points and dense noise
        for idx in true_sparse_indices:
            df.at[idx, 'point_type'] = 'sparse'
        for idx in dense_noise_indices:
            df.at[idx, 'point_type'] = 'dense_noise'
        
    # --- Handle dense noise points as transitional areas ---
    transitional_areas = []  # Store transitional areas for sorting
    if len(dense_noise_indices) >= 3:
        update_progress('clustering', 'processing', f'Analyzing {len(dense_noise_indices)} potential transitional areas...')
        print("\nAnalyzing dense noise points as transitional areas...")
        dense_noise_points = scaled_embeddings[dense_noise_indices]
        
        # Use HDBSCAN to find subgroups within transitional areas
        min_size = max(3, len(dense_noise_points) // 10)
        print(f"Attempting to identify transitional area subgroups with min_size={min_size}")
        
        hdb_dense = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(2, min_size // 2),
            cluster_selection_epsilon=0.3,
            cluster_selection_method='leaf'
        )
        dense_labels = hdb_dense.fit_predict(dense_noise_points)
        
        # Count potential transitional areas
        unique_dense_labels = set(dense_labels) - {-1}
        n_transitional = len(unique_dense_labels)
        print(f"Found {n_transitional} distinct transitional areas")
        
        # First get all transitional points, including scattered ones
        all_transitional_points = {}
        # Count sizes first
        label_sizes = {}
        for label in dense_labels:
            if label != -1:
                label_sizes[label] = label_sizes.get(label, 0) + 1

        # Then collect points with their pre-calculated sizes
        for i, label in enumerate(dense_labels):
            idx = dense_noise_indices[i]
            if label != -1:  # Regular transitional area
                if label not in all_transitional_points:
                    all_transitional_points[label] = {'indices': [], 'size': label_sizes[label]}
                all_transitional_points[label]['indices'].append(idx)
            else:  # Scattered points
                label_key = 'scattered'
                if label_key not in all_transitional_points:
                    all_transitional_points[label_key] = {'indices': [], 'size': 0}
                all_transitional_points[label_key]['indices'].append(idx)
                all_transitional_points[label_key]['size'] += 1
        
        # Sort transitional areas by size and create insights
        # Filter out areas that are too small and sort by size
        min_area_size = 3  # Minimum size for a valid transitional area
        valid_areas = [(k, v) for k, v in all_transitional_points.items() 
                     if k != 'scattered' and v['size'] >= min_area_size]
        sorted_areas = sorted(valid_areas, key=lambda x: x[1]['size'], reverse=True)
        
        # Add regular transitional areas to insights
        total_areas = len(sorted_areas)
        for area_idx, (label, area_info) in enumerate(sorted_areas):
            update_progress('clustering', 'processing', f'Analyzing transitional area {area_idx + 1} of {total_areas} ({area_info["size"]} patents)...')
            area_patents = df.iloc[area_info['indices']]
            description = analyze_patent_group(area_patents, 'transitional', label)
            area_number = area_idx + 1  # 1-based numbering for display
            
            # Create label without duplicate size info
            area_label = f"Transitional Area {area_number}"
            transitional_areas.append({
                'label': area_label,
                'indices': area_info['indices'],
                'size': area_info['size'],
                'patents': area_patents,
                'description': description
            })
            area_insight = {
                'type': 'transitional',
                'id': area_idx + 1,  # Store as 1-based ID
                'size': area_info['size'],
                'label': f"{area_label} ({area_info['size']} patents)",
                'description': description
            }
            cluster_insights.append(area_insight)
        
        # Handle scattered points by analyzing them individually
        if 'scattered' in all_transitional_points:
            scattered_indices = all_transitional_points['scattered']['indices']
            if len(scattered_indices) > 0:
                print(f"\nAnalyzing {len(scattered_indices)} scattered points...")
                scattered_points = scaled_embeddings[scattered_indices]
                
                # Calculate distances to nearest cluster and transitional area
                distances_to_clusters = []
                distances_to_transitional = []
                
                print("\nDistance analysis for each scattered point:")
                point_counter = 0
                
                # First calculate all distances
                for point in scattered_points:
                    point = point.reshape(1, -1)
                    # Distance to nearest cluster
                    if len(cluster_centers) > 0:
                        dist_cluster = NearestNeighbors(n_neighbors=1).fit(cluster_centers).kneighbors(point)[0][0][0]
                    else:
                        dist_cluster = float('inf')
                        
                    # Distance to nearest transitional area (excluding scattered points)
                    if len(dense_noise_points) > 0:
                        # Get only the transitional area points (excluding scattered points)
                        transitional_points = []
                        for i, point_idx in enumerate(dense_noise_indices):
                            if point_idx not in scattered_indices:
                                transitional_points.append(dense_noise_points[i])
                        
                        if transitional_points:
                            transitional_points = np.array(transitional_points)
                            nbrs_trans = NearestNeighbors(n_neighbors=1).fit(transitional_points)
                            dist_trans = nbrs_trans.kneighbors(point.reshape(1, -1))[0][0][0]
                        else:
                            dist_trans = float('inf')
                    else:
                        dist_trans = float('inf')
                    
                        # Store distances for ratio calculation
                    distances_to_clusters.append(dist_cluster)
                    distances_to_transitional.append(dist_trans)

                total_classified_as_gaps = 0
                total_classified_as_transitional = 0
                
                # Use more aggressive thresholds for scattered points
                cluster_distance_threshold = np.percentile(distances_to_clusters, 35)  # Even more lenient
                transitional_distance_threshold = np.percentile(distances_to_transitional, 35)  # Even more lenient
                
                print(f"\nClassification thresholds:")
                print(f"- Cluster distance threshold: {cluster_distance_threshold:.3f}")
                print(f"- Transitional distance threshold: {transitional_distance_threshold:.3f}")
                
                # Classify scattered points
                for idx, (dist_c, dist_t) in zip(scattered_indices, zip(distances_to_clusters, distances_to_transitional)):
                    # 1. Check absolute distances with more lenient thresholds
                    cluster_dist_threshold = np.percentile(distances_to_clusters, 60)  # Use 60th percentile
                    trans_dist_threshold = np.percentile(distances_to_transitional, 60)  # Use 60th percentile
                    
                    # Point is isolated if it's farther than median distance from both clusters and transitional areas
                    is_isolated = (dist_c > cluster_dist_threshold or dist_t > trans_dist_threshold)
                    
                    # 2. Calculate isolation based on absolute difference rather than ratio
                    isolation_diff = dist_t - dist_c  # Positive means farther from transitional areas
                    is_relatively_isolated = isolation_diff > 0  # Any positive difference counts
                    
                    # 3. Simplified region formation check
                    nearby_transitional = sum(1 for d in distances_to_transitional if d < trans_dist_threshold)
                    nearby_clusters = sum(1 for d in distances_to_clusters if d < cluster_dist_threshold)
                    
                    # Point forms new region if it has any cluster neighbors
                    forms_new_region = nearby_clusters > 0
                    
                    # Classification decision and immediate DataFrame update
                    # More lenient classification - if the point is isolated OR relatively isolated, mark as gap
                    if is_isolated or is_relatively_isolated:
                        true_sparse_indices.append(idx)
                        df.at[idx, 'point_type'] = 'sparse'  # Immediately update DataFrame
                        total_classified_as_gaps += 1
                    else:
                        dense_noise_indices.append(idx)
                        df.at[idx, 'point_type'] = 'dense_noise'  # Immediately update DataFrame
                        total_classified_as_transitional += 1
                        
                print(f"\nFinal classification summary for scattered points:")
                print(f"- Total scattered points: {len(scattered_indices)}")
                print(f"- Classified as underexplored areas: {total_classified_as_gaps}")
                print(f"- Classified as transitional: {total_classified_as_transitional}")
                if total_classified_as_gaps == 0:
                    print("\nWarning: No scattered points were classified as underexplored areas!")
                    print("Possible reasons:")
                    print("1. Distance thresholds may be too high")
                    print("2. Relative distance ratio may be too strict")
                    print("3. Nearby points criterion may be too restrictive")
                
                if total_classified_as_transitional > 0:
                    # Create a transitional area for scattered points
                    scattered_transitional_patents = df.iloc[dense_noise_indices[-total_classified_as_transitional:]]
                    description = analyze_patent_group(scattered_transitional_patents, 'transitional', 'scattered')
                    area_number = len(transitional_areas) + 1  # 1-based numbering for display
                    
                    # Add to transitional areas
                    area_label = f"Transitional Area {area_number}"
                    transitional_areas.append({
                        'label': area_label,
                        'indices': dense_noise_indices[-total_classified_as_transitional:],
                        'size': total_classified_as_transitional,
                        'patents': scattered_transitional_patents,
                        'description': description
                    })
                    
                    # Add to insights
                    area_insight = {
                        'type': 'transitional',
                        'id': -1,  # Special ID for scattered points
                        'size': total_classified_as_transitional,
                        'label': f"{area_label} ({total_classified_as_transitional} patents)",
                        'description': description
                    }
                    cluster_insights.append(area_insight)
                
                print(f"\nFinal classification summary for scattered points:")
                print(f"True underexplored areas identified: {len(true_sparse_indices)}")
                print(f"Transitional areas identified: {len(dense_noise_indices)}")
                if len(true_sparse_indices) > 0:
                    print(f"Underexplored area ratio: {len(true_sparse_indices)/len(noise_points):.2%}")
                    print("\nUnderexplored Area Criteria Used:")
                    print("1. Density Isolation: Significantly lower density than nearest cluster")
                    print("2. Spatial Isolation: Far from both clusters and other points")
                    print("3. Structural Stability: Forms stable sparse regions with neighbors")
        
        # Update point types in DataFrame for sparse points and dense noise
        for idx in true_sparse_indices:
            df.at[idx, 'point_type'] = 'sparse'
        for idx in dense_noise_indices:
            df.at[idx, 'point_type'] = 'dense_noise'
        
    # --- Analyze underexplored areas ---
    if len(true_sparse_indices) > 0:
        update_progress('clustering', 'processing', f'Analyzing {len(true_sparse_indices)} potential underexplored areas...')
        print(f"\nProcessing {len(true_sparse_indices)} underexplored areas...")
        sparse_patents = df.iloc[true_sparse_indices]
        sparse_points = scaled_embeddings[true_sparse_indices]
        
        # Ensure points are marked as sparse in the DataFrame
        df.loc[true_sparse_indices, 'point_type'] = 'sparse'
        
        # More lenient subclustering parameters for underexplored areas
        min_subcluster_size = max(2, min(5, len(true_sparse_indices) // 10))  # More lenient minimum size
        sparse_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_subcluster_size,
            min_samples=1,  # Most lenient possible
            cluster_selection_epsilon=0.8,  # Even more lenient
            cluster_selection_method='leaf',  # Changed to leaf for finer subcluster detection
            metric='euclidean'
        )
        sparse_labels = sparse_clusterer.fit_predict(sparse_points)
        
        # Collect innovation subclusters for sorting
        innovation_subclusters = []
        for label in set(sparse_labels):
            subcluster_mask = sparse_labels == label
            subcluster_patents = sparse_patents[subcluster_mask]
            subcluster_size = len(subcluster_patents)
            
            # Accept all subclusters, even single points
            description = analyze_patent_group(subcluster_patents, 'innovation_subcluster', label)
            innovation_subclusters.append({
                'label': label,
                'size': subcluster_size,
                'patents': subcluster_patents,
                'description': description
            })
        
        # Sort innovation subclusters by size in descending order
        innovation_subclusters.sort(key=lambda x: x['size'], reverse=True)
        
        # Add sorted innovation subclusters to insights
        total_subclusters = len(innovation_subclusters)
        for idx, subcluster in enumerate(innovation_subclusters):
            update_progress('clustering', 'processing', f'Analyzing underexplored area opportunity {idx + 1} of {total_subclusters} ({subcluster["size"]} patents)...')
            cluster_insights.append({
                'type': 'innovation_subcluster',
                'id': idx + 1,  # Store as 1-based ID
                'size': subcluster['size'],
                'label': f"Underexplored Area {idx + 1}",
                'description': subcluster['description']
            })
    else:
        cluster_insights.append({
            'type': 'innovation_subcluster',
            'id': -1,
            'size': 0,
            'label': 'No Underexplored Areas',
            'description': 'No significant underexplored areas were detected in this technology space.'
        })

    update_progress('visualization', 'processing', 'Creating interactive plot...')
    
    # Create Plotly figure with clusters
    # Ensure all points are properly categorized
    unassigned_mask = df['point_type'] == 'unassigned'
    if any(unassigned_mask):
        print(f"Warning: {sum(unassigned_mask)} points remain unassigned")
        df.loc[unassigned_mask, 'point_type'] = 'cluster'  # Default unassigned to clusters
    
    # Separate points into three categories: clusters, underexplored areas, and dense noise
    cluster_mask = df['point_type'] == 'cluster'
    innovation_gaps_mask = df['point_type'] == 'sparse'
    dense_noise_mask = df['point_type'] == 'dense_noise'
    
    # Create hover text for all points
    hover_text = []
    # Create mapping for underexplored area points to their numbers
    innovation_gap_map = {}
    
    # Map underexplored areas using the analyzed subclusters to ensure consistent numbering
    if len(true_sparse_indices) > 0:
        for idx, subcluster in enumerate(innovation_subclusters, 1):
            for patent in subcluster['patents'].index:
                innovation_gap_map[patent] = idx
    
    # Create mapping for transitional areas
    transitional_area_map = {}
    for area_idx, area in enumerate(transitional_areas):
        for idx in area['indices']:
            transitional_area_map[idx] = {'number': area_idx + 1}

    # Generate hover text for each point
    for idx, row in df.iterrows():
        point_info = ""
        if row['point_type'] == 'sparse':
            gap_number = innovation_gap_map.get(idx)
            if gap_number:
                point_info = f"<br><b>Region:</b> Underexplored Area {gap_number}"
            else:
                point_info = "<br><b>Region:</b> Potential Innovation Area"
        elif row['point_type'] == 'dense_noise':
            area_info = transitional_area_map.get(idx)
            if area_info:
                point_info = f"<br><b>Region:</b> Transitional Area {area_info['number']}"
            else:
                # This is a scattered transitional point
                point_info = f"<br><b>Region:</b> Transitional Area {len(transitional_areas)} (Scattered)"
        else:
            point_info = f"<br><b>Cluster:</b> {int(row['cluster']) + 1}"  # Cluster IDs are still 0-based in the DataFrame
            
        text = (
            f"<b>{row['title']}</b><br><br>"
            f"<b>By:</b> {row['assignee']} ({row['year']})<br>"
            f"{point_info}<br><br>"
            f"<b>Abstract:</b><br>{row['abstract']}"
        )
        hover_text.append(text)

    # Create three separate traces: clusters, underexplored areas, and dense noise points
    cluster_trace = go.Scatter3d(
        x=df[cluster_mask]['x'],
        y=df[cluster_mask]['y'],
        z=df[cluster_mask]['z'],
        mode='markers',
        marker=dict(
            size=6,
            color=clusters[cluster_mask] + 1,  # Add 1 to shift cluster numbers from 0-based to 1-based
            colorscale='Viridis',
            opacity=0.5,
            showscale=True,
            colorbar=dict(
                title="Clusters",
                ticktext=[f"Cluster {i+1}" for i in range(n_clusters)],  # Custom tick labels
                tickvals=list(range(1, n_clusters + 1)),  # Values to match the 1-based cluster numbers
                tickmode="array",
                tickfont=dict(size=10),
                titlefont=dict(size=10)
            )
        ),
        text=[hover_text[i] for i in range(len(hover_text)) if cluster_mask[i]],
        hoverinfo='text',
        name='Clusters',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        ),
        customdata=[df['link'].tolist()[i] for i in range(len(df)) if cluster_mask[i]]
    )

    innovation_gaps_trace = go.Scatter3d(
        x=df[innovation_gaps_mask]['x'],
        y=df[innovation_gaps_mask]['y'],
        z=df[innovation_gaps_mask]['z'],
        mode='markers',
        marker=dict(
            size=6,  # Same size as other points
            color='rgb(255, 0, 0)',  # Pure bright red
            symbol='diamond',
            opacity=1.0,  # Full opacity for visibility
            line=dict(
                color='white',
                width=1  # Thinner border to match other points
            )
        ),
        text=[hover_text[i] for i in range(len(hover_text)) if innovation_gaps_mask[i]],
        hoverinfo='text',
        name='Underexplored Areas',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        ),
        customdata=[df['link'].tolist()[i] for i in range(len(df)) if innovation_gaps_mask[i]]
    )

    dense_noise_trace = go.Scatter3d(
        x=df[dense_noise_mask]['x'],
        y=df[dense_noise_mask]['y'],
        z=df[dense_noise_mask]['z'],
        mode='markers',
        marker=dict(
            size=6,  # Same size as other points
            color='rgb(255, 165, 0)',  # Orange for transitional areas
            symbol='circle',
            opacity=0.7,  # Less opacity to make gaps more visible
            line=dict(
                color='white',
                width=1  # Thin border
            )
        ),
        text=[hover_text[i] for i in range(len(hover_text)) if dense_noise_mask[i]],
        hoverinfo='text',
        name='Transitional Areas',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        ),
        customdata=[df['link'].tolist()[i] for i in range(len(df)) if dense_noise_mask[i]]
    )

    fig = go.Figure(data=[cluster_trace, innovation_gaps_trace, dense_noise_trace])
    
    # Update layout
    fig.update_layout(
        title="Patent Technology Landscape",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.8, y=1.8, z=1.8)  # Slightly further out for better overview
            ),
            aspectmode='cube'  # Force equal scaling
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        template="plotly_dark",
        hoverlabel_align='left',
        hoverdistance=100,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.7)",  # Darker background for better contrast
            font=dict(
                color="white",
                size=12
            ),
            itemsizing='constant'  # Keep legend marker sizes consistent
        )
    )
    
    # Configure hover behavior
    fig.update_traces(
        hovertemplate='%{text}<extra></extra>',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="Arial"
        )
    )
    
    update_progress('visualization', 'processing', 'Finalizing visualization...')
    
    return {
        'plot': fig.to_json(),
        'insights': cluster_insights
    }

def generate_analysis(prompt, cluster_insights):
    """Generate analysis using OpenAI's GPT API with retries and validation"""
    try:
        # Add system context
        messages = [
            {
                "role": "system",
                "content": "You are an expert patent analyst specializing in technology landscapes and innovation opportunities."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        analysis = response.choices[0].message['content']
        
        # Validate that analysis references valid areas
        area_pattern = r'(?:Cluster|Transitional Area|Underexplored Area)\s+(\d+)'
        referenced_areas = set(int(num) for num in re.findall(area_pattern, analysis))
        
        # Extract valid area numbers from insights
        valid_areas = set()
        for insight in cluster_insights:
            if insight['id'] > 0:  # Skip special IDs like -1
                valid_areas.add(insight['id'])
        
        # Check if all referenced areas are valid
        invalid_areas = referenced_areas - valid_areas
        if invalid_areas:
            print(f"Warning: Analysis references invalid areas: {invalid_areas}")
            return "Error: Unable to generate valid analysis. Please try again."
            
        return analysis
        
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return "Error generating innovation analysis. Please try again."

def analyze_innovation_opportunities(cluster_insights):
    """
    Analyze relationships between different areas to identify potential innovation opportunities.
    Returns focused analysis of three key innovation gaps between existing technology areas.
    """
    # Extract cluster numbers and validate
    cluster_nums = set()
    transitional_nums = set()
    underexplored_nums = set()
    
    # Parse and validate cluster numbers with explicit error checking
    for insight in cluster_insights:
        area_type = insight.get('type', '')
        area_id = insight.get('id', -1)
        
        if area_id < 0 and area_type != 'cluster':
            continue
            
        if area_type == 'cluster':
            cluster_nums.add(area_id)
        elif area_type == 'transitional':
            transitional_nums.add(area_id)
        elif area_type == 'innovation_subcluster':
            if area_id >= 1:  # Skip the "No underexplored areas" entry
                underexplored_nums.add(area_id)

    # Format areas list with validation
    def format_area_list(area_nums):
        return f"Areas {', '.join(str(n) for n in sorted(area_nums))}" if area_nums else "None identified"

    # Only generate analysis if we have areas to analyze
    if not any([cluster_nums, transitional_nums, underexplored_nums]):
        return "No distinct areas found. Try broadening search terms or increasing patent count."

    # Create descriptions list
    descriptions = []
    for insight in cluster_insights:
        if insight.get('description'):
            area_type = insight.get('type', '')
            area_id = int(insight.get('id', -1))  # 1-based IDs
            if area_type == 'cluster':
                desc = f"C{area_id}:{insight['description']}"
            elif area_type == 'transitional':
                desc = f"T{area_id}:{insight['description']}"
            elif area_type == 'innovation_subcluster' and insight['id'] >= 1:
                desc = f"U{area_id}:{insight['description']}"
            else:
                continue
            descriptions.append(desc)
    
    # Format descriptions as a string with newlines
    descriptions_text = '\n'.join(descriptions)

    prompt = f"""Available Areas:
Clusters: {format_area_list(cluster_nums)}
Transitional Areas: {format_area_list(transitional_nums)} 
Underexplored Areas: {format_area_list(underexplored_nums)}
Area Descriptions:
{descriptions_text}
Analyze the most promising innovation opportunities. For each opportunity:
1. Identify two technologically complementary areas (e.g. "Cluster 1 + Transitional Area 2")
2. Focus on specific technical capabilities that could be combined
3. Aim for practical, near-term innovations
Provide 3 opportunities, formatted as:
Opportunity N:
[Area 1] + [Area 2]
- Gap: Specific technical capability missing between these areas
- Solution: Concrete technical approach using existing methods
- Impact: Clear technical or market advantage gained
Prioritize:
- Technical feasibility over speculative concepts
- Cross-domain applications with clear synergies
- Opportunities that build on existing technology strengths"""

    # Get analysis from LLM
    response = generate_analysis(prompt, cluster_insights)
    return response

def update_progress(step, status='processing', message=None):
    """Update progress through the progress queue"""
    progress_queue = app.config['PROGRESS_QUEUE']
    data = {
        'step': step,
        'status': status
    }
    if message:
        data['message'] = message
    progress_queue.put(data)

# Add error handlers right before the routes
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found - please check the URL and try again'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error occurred'}), 500

# Add index route before other routes
@app.route('/')
def home():
    """Home page route - check for existing visualizations"""
    # Check if we have any visualization data
    has_visualization = False
    
    # If this is a new session, check for existing visualizations
    if not session.get('viz_file') or not os.path.exists(session.get('viz_file')):
        # Define directories based on environment
        if IS_DOCKER:
            # Use Linux-style paths for Docker
            base_dir = "/home/user/app"
            data_dir = os.path.join(base_dir, 'data', 'visualizations')
            temp_dir = '/tmp'
        else:
            # Use platform-independent paths for local development
            base_dir = os.getcwd()
            data_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))
            temp_dir = tempfile.gettempdir()
        
        # Look for any visualization files in both directories
        print(f"Checking for existing visualizations in data dir: {data_dir}")
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.startswith("patent_viz_") and filename.endswith(".json"):
                    print(f"Found visualization in data dir: {filename}")
                    has_visualization = True
                    break
        
        # Also check temp directory
        if not has_visualization and os.path.exists(temp_dir):
            print(f"Checking for existing visualizations in temp dir: {temp_dir}")
            for filename in os.listdir(temp_dir):
                if filename.startswith("patent_viz_") and filename.endswith(".json"):
                    print(f"Found visualization in temp dir: {filename}")
                    has_visualization = True
                    break
    else:
        print(f"Session already has visualization file: {session.get('viz_file')}")
        has_visualization = True
    
    print(f"Has existing visualization: {has_visualization}")
    return render_template('index.html', has_existing_visualization=has_visualization)

@app.route('/progress')
def get_progress():
    """Server-sent events endpoint for progress updates"""
    progress_queue = app.config['PROGRESS_QUEUE']
    def generate():
        connection_active = True
        while connection_active:
            try:
                data = progress_queue.get(timeout=10) # Reduced timeout for more responsive updates
                if data == 'DONE':
                    yield f"data: {json.dumps({'step': 'complete', 'status': 'done'})}\n\n"
                    connection_active = False
                else:
                    yield f"data: {json.dumps(data)}\n\n"
            except Empty:
                # Send a keep-alive message
                yield f"data: {json.dumps({'step': 'alive', 'status': 'processing'})}\n\n"
                continue
            
            # Ensure the data is sent immediately
            if hasattr(generate, 'flush'):
                generate.flush()

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'Content-Type': 'text/event-stream',
        'X-Accel-Buffering': 'no'  # Disable buffering for nginx
    })

@app.route('/search', methods=['POST']) 
def search():
    progress_queue = app.config['PROGRESS_QUEUE']
    while not progress_queue.empty():
        progress_queue.get_nowait()
    keywords = request.form.get('keywords', '')
    if not keywords:
        return jsonify({'error': 'Please enter search keywords'})
    
    print(f"\nProcessing search request for keywords: {keywords}")
    
    try:
        # Use existing session ID, never create new one here
        session_id = session.get('id')
        if not session_id:
            return jsonify({'error': 'Invalid session'})
            
        data_path = session.get('viz_file')
        temp_path = session.get('temp_viz_file')
        if not data_path or not temp_path:
            return jsonify({'error': 'Invalid session paths'})
        
        # Clear any existing progress updates
        while not progress_queue.empty():
            progress_queue.get_nowait()
            
        # Initial progress update
        update_progress('search', 'processing', 'Starting patent search...')
        patents = search_patents(keywords)
        if not patents:
            update_progress('search', 'error', 'No patents found')
            progress_queue.put('DONE')
            return jsonify({'error': 'No patents found or an error occurred'})
        
        # Generate visualization and insights
        update_progress('visualization', 'Creating visualization...')
        viz_data = create_3d_visualization(patents)
        if not viz_data or not viz_data.get('plot'):
            progress_queue.put('DONE')
            return jsonify({'error': 'Error creating visualization'})
            
        # Generate innovation analysis from insights
        innovation_analysis = analyze_innovation_opportunities(viz_data['insights'])
        
        # Store innovation analysis in visualization data for persistence
        viz_data['innovation_analysis'] = innovation_analysis
        
        # Save visualization data to persistent storage
        data_path = session['viz_file']
        temp_path = session['temp_viz_file']
        
        # Save to persistent storage
        print(f"Saving visualization to: {data_path}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            with open(data_path, 'w') as f:
                json.dump(viz_data, f)
                f.flush()
                os.fsync(f.fileno())
            print(f"Successfully saved visualization to {data_path}")
        except Exception as e:
            print(f"Error saving visualization to {data_path}: {e}")
            
        # Save to temp storage
        print(f"Saving temp copy to: {temp_path}")
        try:
            # Ensure temp directory exists
            temp_dir = os.path.dirname(temp_path)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
                
            with open(temp_path, 'w') as f:
                json.dump(viz_data, f)
            print(f"Successfully saved temp copy to {temp_path}")
        except Exception as e:
            print(f"Error saving temp copy to {temp_path}: {e}")
            
        session.modified = True
        
        # Only store analysis in session since it's smaller
        session['last_analysis'] = innovation_analysis
        
        # Final progress update
        update_progress('complete', 'Analysis complete!')
        progress_queue.put('DONE')
        
        return jsonify({
            'visualization': viz_data['plot'],
            'insights': viz_data['insights'],
            'innovationAnalysis': innovation_analysis
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        progress_queue.put('DONE')
        return jsonify({'error': str(e)})

@app.route('/download_plot')
def download_plot():
    try:
        # Add debug logging
        print("\nDownload Plot Debug Info:")
        print(f"Session ID: {session.get('id')}")
        print(f"Session data: {dict(session)}")
        
        # Use the global Docker environment variable
        print(f"Running in Docker: {IS_DOCKER}")
        
        # Get paths from session
        data_path = session.get('viz_file')
        temp_path = session.get('temp_viz_file')
        
        # Log paths and check if they exist
        print(f"Data path: {data_path}")
        if data_path:
            data_exists = os.path.exists(data_path)
            print(f"Data path exists: {data_exists}")
            if not data_exists:
                # Debug directory contents
                parent_dir = os.path.dirname(data_path)
                print(f"Parent directory ({parent_dir}) exists: {os.path.exists(parent_dir)}")
                if os.path.exists(parent_dir):
                    print(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
        
        print(f"Temp path: {temp_path}")
        if temp_path:
            temp_exists = os.path.exists(temp_path)
            print(f"Temp path exists: {temp_exists}")
            if not temp_exists:
                # Debug temp directory
                temp_dir = os.path.dirname(temp_path)
                print(f"Temp directory ({temp_dir}) exists: {os.path.exists(temp_dir)}")
                if os.path.exists(temp_dir):
                    print(f"Contents of {temp_dir}: {os.listdir(temp_dir)}")
        
        # Try both locations
        viz_file = None
        if data_path and os.path.exists(data_path):
            viz_file = data_path
            print(f"Using primary data path: {viz_file}")
        elif temp_path and os.path.exists(temp_path):
            viz_file = temp_path
            print(f"Using temp path: {viz_file}")
            # Copy to persistent storage if only in temp
            try:
                with open(temp_path, 'r') as f:
                    viz_data = json.load(f)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w') as f:
                    json.dump(viz_data, f)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"Copied temp file to persistent storage: {data_path}")
            except Exception as e:
                print(f"Error copying from temp to persistent storage: {e}")
        else:
            # If no visualization file for current session, try to find the most recent one
            print("No visualization file found for current session. Searching for most recent visualization...")
            
            # Determine directory paths based on environment
            if IS_DOCKER:
                # Use Linux-style paths for Docker
                base_dir = "/home/user/app"
                data_parent_dir = os.path.join(base_dir, 'data', 'visualizations')
                temp_parent_dir = '/tmp'
            else:
                # Use platform-independent paths for local development
                base_dir = os.getcwd()
                data_parent_dir = os.path.normpath(os.path.join(base_dir, 'data', 'visualizations'))
                temp_parent_dir = tempfile.gettempdir()
            
            most_recent_file = None
            most_recent_time = 0
            
            # Check data directory first
            if os.path.exists(data_parent_dir):
                print(f"Checking data directory: {data_parent_dir}")
                for filename in os.listdir(data_parent_dir):
                    if filename.startswith("patent_viz_") and filename.endswith(".json"):
                        file_path = os.path.join(data_parent_dir, filename)
                        file_time = os.path.getmtime(file_path)
                        print(f"Found file: {file_path}, modified: {datetime.fromtimestamp(file_time)}")
                        if file_time > most_recent_time:
                            most_recent_time = file_time
                            most_recent_file = file_path
            
            # Then check temp directory
            if os.path.exists(temp_parent_dir):
                print(f"Checking temp directory: {temp_parent_dir}")
                for filename in os.listdir(temp_parent_dir):
                    if filename.startswith("patent_viz_") and filename.endswith(".json"):
                        file_path = os.path.join(temp_parent_dir, filename)
                        file_time = os.path.getmtime(file_path)
                        print(f"Found file: {file_path}, modified: {datetime.fromtimestamp(file_time)}")
                        if file_time > most_recent_time:
                            most_recent_time = file_time
                            most_recent_file = file_path
            
            if most_recent_file:
                print(f"Found most recent visualization file: {most_recent_file}")
                viz_file = most_recent_file
                
                # Update the session with this file
                try:
                    # Copy to this session's files
                    with open(most_recent_file, 'r') as f:
                        viz_data = json.load(f)
                    
                    # Save to the current session's data path
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    with open(data_path, 'w') as f:
                        json.dump(viz_data, f)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Also save to temp path
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    with open(temp_path, 'w') as f:
                        json.dump(viz_data, f)
                    
                    print(f"Copied most recent visualization to current session's files")
                    viz_file = data_path  # Use the new file for this session
                    
                    # Update session paths
                    session['viz_file'] = data_path
                    session['temp_viz_file'] = temp_path
                    session.modified = True
                except Exception as e:
                    print(f"Error copying most recent visualization to current session: {e}")
            else:
                print("No visualization files found in either location")
                return jsonify({'error': 'No visualizations found. Please run a new search.'}), 404  # Return 404 status code
            
        # Continue with existing download code...
        try:
            print(f"Reading visualization file: {viz_file}")
            with open(viz_file, 'r') as f:
                viz_data = json.load(f)
                print(f"Visualization data keys: {viz_data.keys()}")
                plot_data = viz_data.get('plot')
                if not plot_data:
                    print("No plot data found in visualization file")
                    # Check what's actually in the file
                    print(f"Visualization data contains: {viz_data.keys()}")
                    return jsonify({'error': 'Invalid plot data - missing plot field'}), 404
                print("Successfully loaded plot data")
        except json.JSONDecodeError as je:
            print(f"JSON decode error when reading visualization file: {je}")
            # Try to read raw file
            try:
                with open(viz_file, 'r') as f:
                    raw_content = f.read()
                print(f"Raw file content (first 200 chars): {raw_content[:200]}")
            except Exception as e2:
                print(f"Error reading raw file: {e2}")
            return jsonify({'error': f'Corrupt visualization data: {str(je)}'}), 500
        except Exception as e:
            print(f"Error reading visualization file: {e}")
            return jsonify({'error': f'Failed to read visualization data: {str(e)}'}), 500
        
        # Create a temporary file for the HTML
        try:
            print("Creating temporary HTML file...")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                # Write the HTML content
                html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Patent Technology Landscape</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plot"></div>
    <script>
        var plotData = %s;
        Plotly.newPlot('plot', plotData.data, plotData.layout);
    </script>
</body>
</html>
                """ % plot_data
                
                f.write(html_content)
                temp_html_path = f.name
                print(f"Created temporary HTML file at: {temp_html_path}")
            
            print("Sending file to user...")
            return send_file(
                temp_html_path,
                as_attachment=True,
                download_name='patent_landscape.html',
                mimetype='text/html'
            )
        except Exception as e:
            print(f"Error creating or sending HTML file: {e}")
            return jsonify({'error': f'Failed to generate plot file: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in download_plot: {e}")
        return jsonify({'error': f'Failed to process download request: {str(e)}'}), 500

@app.route('/download_insights')
def download_insights():
    """Download the latest insights as a PDF file"""
    try:
        # Check if session exists
        if not session.get('id'):
            return jsonify({'error': 'No active session found. Please run a new search.'})

        viz_file = session.get('viz_file')
        analysis = session.get('last_analysis')
        print(f"Visualization file path from session: {viz_file}")
        print(f"Analysis data available: {bool(analysis)}")
        
        if not viz_file:
            print("No visualization file path found in session")
            return jsonify({'error': 'No insights available - missing file path'})
            
        if not os.path.exists(viz_file):
            print(f"Visualization file does not exist at path: {viz_file}")
            return jsonify({'error': 'No insights available - file not found'})
        
        try:
            print(f"Reading visualization file: {viz_file}")
            with open(viz_file, 'r') as f:
                viz_data = json.load(f)
                insights = viz_data.get('insights')
                if not insights:
                    print("No insights found in visualization file")
                    return jsonify({'error': 'Invalid insights data - missing insights field'})
                print(f"Successfully loaded insights data with {len(insights)} insights")
                
                # If no analysis in session, try to get it from the visualization data
                if not analysis and 'innovation_analysis' in viz_data:
                    analysis = viz_data.get('innovation_analysis')
                    print("Retrieved innovation analysis from visualization file")
        except Exception as e:
            print(f"Error reading visualization file: {e}")
            return jsonify({'error': f'Failed to load insights: {str(e)}'})
        
        # Create a PDF in memory
        print("Creating PDF in memory...")
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkblue
        )
        opportunity_style = ParagraphStyle(
            'OpportunityStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=5,
            leftIndent=20,
            firstLineIndent=0
        )
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=5,
            leftIndent=40,
            firstLineIndent=-20
        )
        
        # Build the document
        try:
            print("Building PDF document structure...")
            story = []
            story.append(Paragraph("Patent Technology Landscape Analysis", title_style))
            
            # Add innovation analysis first if available
            if analysis:
                print("Adding innovation opportunities analysis...")
                story.append(Paragraph("Innovation Opportunities Analysis", heading_style))
                
                # Format the innovation analysis for better readability
                # Look for opportunity patterns in the text
                analysis_parts = []
                
                # Split by "Opportunity" keyword to identify sections
                import re
                opportunity_pattern = r'Opportunity\s+\d+:'
                opportunity_matches = re.split(opportunity_pattern, analysis)
                
                # First part may be an introduction
                if opportunity_matches and opportunity_matches[0].strip():
                    story.append(Paragraph(opportunity_matches[0].strip(), normal_style))
                    story.append(Spacer(1, 10))
                
                # Process each opportunity section
                for i in range(1, len(opportunity_matches)):
                    opp_text = opportunity_matches[i].strip()
                    opp_title = f"Opportunity {i}:"
                    story.append(Paragraph(opp_title, subheading_style))
                    
                    # Process sections like [Area 1] + [Area 2], Gap, Solution, Impact
                    opp_lines = opp_text.split('\n')
                    for j, line in enumerate(opp_lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Format the first line (Area combinations) specially
                        if j == 0:
                            story.append(Paragraph(line, opportunity_style))
                        # Look for bullet points (Gap, Solution, Impact)
                        elif line.startswith('-'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                bullet = parts[0].strip('- ')
                                content = parts[1].strip()
                                formatted_line = f"• <b>{bullet}:</b> {content}"
                                story.append(Paragraph(formatted_line, bullet_style))
                            else:
                                story.append(Paragraph(line, bullet_style))
                        else:
                            story.append(Paragraph(line, opportunity_style))
                    
                    # Add space between opportunities
                    story.append(Spacer(1, 15))
                
                # If we couldn't parse the format, just add the raw text
                if len(opportunity_matches) <= 1:
                    story.append(Paragraph(analysis, normal_style))
                
                # Add separator
                story.append(Spacer(1, 20))
            
            # Add clusters
            print("Adding technology clusters section...")
            story.append(Paragraph("Technology Clusters", heading_style))
            cluster_count = 0
            for insight in insights:
                if insight['type'] == 'cluster':
                    text = f"<b>Cluster {insight['id']}:</b> {insight['description']}"
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 12))
                    cluster_count += 1
            print(f"Added {cluster_count} clusters")
            
            # Add transitional areas
            print("Adding transitional areas section...")
            story.append(Paragraph("Transitional Areas", heading_style))
            trans_count = 0
            for insight in insights:
                if insight['type'] == 'transitional':
                    text = f"<b>Transitional Area {insight['id']}:</b> {insight['description']}"
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 12))
                    trans_count += 1
            print(f"Added {trans_count} transitional areas")
            
            # Add underexplored areas
            print("Adding underexplored areas section...")
            story.append(Paragraph("Underexplored Areas", heading_style))
            underexplored_count = 0
            for insight in insights:
                if insight['type'] == 'innovation_subcluster':
                    text = f"<b>Underexplored Area {insight['id']}:</b> {insight['description']}"
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 12))
                    underexplored_count += 1
            print(f"Added {underexplored_count} underexplored areas")
            
            # Build PDF
            print("Building final PDF document...")
            doc.build(story)
            buffer.seek(0)
            
            print("Sending PDF file to user...")
            return send_file(
                buffer,
                as_attachment=True,
                download_name='patent_insights.pdf',
                mimetype='application/pdf'
            )
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return jsonify({'error': f'Failed to generate PDF file: {str(e)}'})
            
    except Exception as e:
        print(f"Error in download_insights: {e}")
        return jsonify({'error': f'Failed to process download request: {str(e)}'})

@app.teardown_request
def cleanup_temp_files(exception=None):
    """Clean up temporary files when they are no longer needed"""
    try:
        # Only cleanup files that were created in previous sessions
        temp_dir = tempfile.gettempdir()
        current_time = time.time()
        # Look for visualization files that are older than 30 minutes
        for filename in os.listdir(temp_dir):
            if filename.startswith('patent_viz_') and filename.endswith('.json'):
                filepath = os.path.join(temp_dir, filename)
                # Check if file is older than 30 minutes
                if current_time - os.path.getmtime(filepath) > 1800:  # 30 minutes in seconds
                    try:
                        os.remove(filepath)
                        print(f"Cleaned up old temporary file: {filepath}")
                    except Exception as e:
                        print(f"Error cleaning up temporary file: {e}")
    except Exception as e:
        print(f"Error in cleanup: {e}")
        # Don't raise the exception to prevent request handling failures

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)