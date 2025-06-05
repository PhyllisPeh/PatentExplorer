from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
import requests
from datetime import datetime
import os
import json
import openai
import numpy as np
import pickle
from pathlib import Path
import umap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import queue
import threading
import hdbscan
from sklearn.neighbors import NearestNeighbors

load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_PATENTS = 3000  # Increased from 2000 to 5000 for better coverage
MIN_PATENTS_FOR_GAPS = 3000  # Minimum patents needed for reliable gap detection
CACHE_FILE = 'patent_embeddings_cache.pkl'

# Global progress queue for SSE updates
progress_queue = queue.Queue()

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
    Search patents using SerpApi's Google Patents API with pagination and generate embeddings
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
                assignee = patent.get('assignee', 'N/A')
                if isinstance(assignee, list) and assignee:
                    assignee = assignee[0]

                # Format title and abstract for embedding
                title = patent.get('title', '').strip()
                abstract = patent.get('snippet', '').strip()
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
                    'link': patent.get('patent_link', '') or patent.get('link', ''),
                    'embedding': embedding
                }
                all_patents.append(formatted_patent)
            
            print(f"Retrieved {len(patents_data)} patents from page {page}")
            
            # Check if there are more pages
            if not response_data.get('serpapi_pagination', {}).get('next'):
                break
                
            page += 1
            
        except Exception as e:
            print(f"Error searching patents: {e}")
            break
    
    # Save final cache state
    save_cache(embedding_cache)
    
    print(f"Total patents retrieved and embedded: {len(all_patents)}")
    return all_patents

def analyze_patent_group(patents, group_type, label):
    """
    Analyze a group of patents using ChatGPT.
    group_type: 'cluster', 'innovation_subcluster', 'transitional'
    """
    titles = "; ".join(patents['title'].tolist()[:3])  # Show top 3 titles
    assignees = ", ".join(patents['assignee'].unique())
    years = f"{patents['year'].min()} - {patents['year'].max()}"
    
    if group_type == 'cluster':
        prompt = f"""Technology Cluster Analysis

Titles: {titles}
Assignees: {assignees}
Years: {years}

Please provide a focused analysis of this technology cluster:
1. Technology focus: What specific technology area do these patents represent? (1-2 short sentences)
2. Current trend: What development trends are visible in this cluster? (1-2 short sentences)
3. Distinctive keywords: What technical terms are unique to this cluster and differentiate it from others? List 4-6 keywords that characterize this specific technology area."""
        
        system_prompt = "You are a patent expert analyzing technology clusters. Focus on what makes this cluster unique and its current development trends."
    elif group_type == 'transitional':
        prompt = f"""Transitional Area Analysis

Titles: {titles}
Assignees: {assignees}
Years: {years}

Please provide a focused analysis of this transitional technology area:
1. Technology focus: What specific technology area do these patents represent? (1-2 short sentences)
2. Emerging trends: What potential new technology directions or cross-domain applications are suggested? (1-2 short sentences)
3. Bridge concepts: What technical terms suggest connections between established technology areas? List 4-6 keywords that highlight potential technology convergence."""
        
        system_prompt = "You are a patent expert analyzing transitional technology areas. Focus on identifying emerging trends and potential technology convergence points."
    else:  # innovation_subcluster
        prompt = f"""Innovation Gap Analysis

Titles: {titles}
Assignees: {assignees}
Years: {years}

Please provide a focused analysis of this innovation opportunity:
1. Technology focus: What specific technology area do these patents represent? (1-2 short sentences)
2. Innovation potential: What unique opportunities exist in this specific area? (1-2 short sentences)
3. Distinctive keywords: What technical terms highlight the innovation potential? List 4-6 keywords that characterize this opportunity area."""
        
        system_prompt = "You are a patent expert analyzing innovation opportunities. Focus on what makes this area unique and its specific innovation potential."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.4
        )
        return response.choices[0]['message']['content']
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return "Analysis generation failed."

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
        print("Innovation gap detection may be less reliable with smaller datasets")
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
    
    # Adjust min_cluster_size to be more aggressive for better cluster separation
    min_cluster_size = max(5, min(30, int(n_points * 0.02)))  # 2% of data, between 5 and 30
    min_samples = max(3, min(15, int(n_points * 0.01)))      # 1% of data, between 3 and 15

    # Dynamically set max_clusters and target_noise based on number of patents
    if n_points < 100:
        max_clusters = 5
        max_retries = 2
        target_noise_ratio = 0.15  # 15% noise for very small datasets
    elif n_points < 200:
        max_clusters = 8
        max_retries = 3
        target_noise_ratio = 0.12  # 12% noise for small datasets
    elif n_points < 500:
        max_clusters = 12
        max_retries = 4
        target_noise_ratio = 0.10  # 10% noise for medium datasets
    else:
        max_clusters = 15
        max_retries = 5
        target_noise_ratio = 0.08  # 8% noise for large datasets

    target_noise = int(n_points * target_noise_ratio)
    print(f"Initial HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}, max_clusters={max_clusters}, max_retries={max_retries}, target_noise={target_noise}")
    retry = 0
    clusters = None
    n_clusters = 0
    n_noise = 0

    while retry < max_retries:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.2,  # More aggressive cluster separation
            cluster_selection_method='eom',  # Excess of Mass method for better cluster identification
            metric='euclidean'
        )
        clusters = hdb.fit_predict(scaled_embeddings)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print(f"\nClustering Statistics (try {retry+1}):")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of patents in sparse regions: {n_noise}")
        print(f"Total number of patents: {len(clusters)}")
        
        # Check for problematic clustering outcomes
        noise_ratio = n_noise / len(clusters)
        avg_cluster_size = (len(clusters) - n_noise) / n_clusters if n_clusters > 0 else float('inf')
        
        print(f"Noise ratio: {noise_ratio:.2%}")
        print(f"Average cluster size: {avg_cluster_size:.1f} patents")
        
        update_progress('clustering', 'processing', 
            f'Optimizing clusters (attempt {retry + 1}/{max_retries}): ' +
            f'Found {n_clusters} clusters with avg size {avg_cluster_size:.1f} patents')
        
        # If one giant cluster or too many tiny clusters
        if n_clusters == 1 and avg_cluster_size > len(clusters) * 0.8:
            print("Single dominant cluster detected, adjusting for better separation...")
            min_cluster_size = max(5, int(min_cluster_size * 0.8))
            min_samples = max(3, int(min_samples * 0.8))
            retry += 1
            continue
            
        # If too many clusters, merge by increasing parameters
        if n_clusters > max_clusters:
            print("Too many clusters, increasing parameters for merging...")
            min_cluster_size = int(min_cluster_size * 1.3)
            min_samples = int(min_samples * 1.2)
            retry += 1
            continue
            
        # If too few meaningful clusters or no sparse regions
        if n_clusters < 2 or n_noise == 0 or noise_ratio < 0.05:
            print("Insufficient cluster separation, adjusting parameters...")
            min_cluster_size = max(3, int(min_cluster_size * 0.7))
            min_samples = max(2, int(min_samples * 0.7))
            retry += 1
            continue
        # Acceptable clustering found
        break
    else:
        print("Max retries reached. Proceeding with last clustering result.")

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
    
    update_progress('clustering', 'processing', 'Identifying technology clusters and innovation gaps...')
    
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
            'id': int(new_id),
            'size': size,
            'label': f"Cluster {new_id}",
            'description': description
        })

    # --- Improved two-stage density analysis for noise points ---
    noise_mask = df['cluster'] == -1
    noise_points = scaled_embeddings[noise_mask]
    noise_indices = df[noise_mask].index
    dense_noise_indices = []  # Initialize empty list for dense noise points
    
    if len(noise_points) >= 3:
        update_progress('clustering', 'processing', f'Analyzing {len(noise_points)} potential innovation gaps...')
        print(f"\nStructural Analysis for Innovation Gap Detection:")
        
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
            
            # Identify structural gaps using multiple criteria
            # 1. Density Isolation: Points with very low density compared to clusters
            # 2. Spatial Isolation: Points far from both clusters and other noise points
            # 3. Structural Stability: Points whose local neighborhood is also sparse
            
            # Calculate isolation scores with adjusted thresholds
            density_isolation = density_ratios < 0.4  # Much more lenient density threshold
            spatial_isolation = cluster_distances > np.percentile(cluster_distances, 60)  # More lenient spatial threshold
            
            # Calculate structural stability with more lenient criteria
            structural_stability = np.zeros(len(noise_points), dtype=bool)
            for i, neighbors in enumerate(local_indices):
                neighbor_densities = local_densities[neighbors]
                # Point is stable if its neighborhood is relatively sparse
                structural_stability[i] = np.mean(neighbor_densities) < np.mean(local_densities)
            
            # First identify potential innovation gaps using basic criteria - need to meet 2 out of 3 criteria
            candidate_sparse_indices = [
                idx for i, idx in enumerate(noise_indices)
                if sum([density_isolation[i], spatial_isolation[i], structural_stability[i]]) >= 2  # Only need 2 out of 3 criteria
            ]
            
            # Start by assuming all non-candidate points are dense noise
            dense_noise_indices = [idx for idx in noise_indices if idx not in candidate_sparse_indices]
            
            # Now we can safely calculate distances between candidates and dense noise points
            min_distance_threshold = np.percentile(cluster_distances, 60)  # 60th percentile of distances
            
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
        print(f"True innovation gaps identified: {len(true_sparse_indices)}")
        print(f"Transitional areas identified: {len(dense_noise_indices)}")
        if len(true_sparse_indices) > 0:
            print(f"Innovation gap ratio: {len(true_sparse_indices)/len(noise_points):.2%}")
            print("\nInnovation Gap Criteria Used:")
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
                'id': int(label),
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
                    if (is_isolated or is_relatively_isolated) and forms_new_region:
                        true_sparse_indices.append(idx)
                        df.at[idx, 'point_type'] = 'sparse'  # Immediately update DataFrame
                        total_classified_as_gaps += 1
                    else:
                        dense_noise_indices.append(idx)
                        df.at[idx, 'point_type'] = 'dense_noise'  # Immediately update DataFrame
                        total_classified_as_transitional += 1
                        
                print(f"\nFinal classification summary for scattered points:")
                print(f"- Total scattered points: {len(scattered_indices)}")
                print(f"- Classified as innovation gaps: {total_classified_as_gaps}")
                print(f"- Classified as transitional: {total_classified_as_transitional}")
                if total_classified_as_gaps == 0:
                    print("\nWarning: No scattered points were classified as innovation gaps!")
                    print("Possible reasons:")
                    print("1. Distance thresholds may be too high")
                    print("2. Relative distance ratio may be too strict")
                    print("3. Nearby points criterion may be too restrictive")
    
    # --- Analyze innovation gaps ---
    if len(true_sparse_indices) > 0:
        update_progress('clustering', 'processing', f'Analyzing {len(true_sparse_indices)} potential innovation opportunities...')
        print(f"\nProcessing {len(true_sparse_indices)} innovation gaps...")
        sparse_patents = df.iloc[true_sparse_indices]
        sparse_points = scaled_embeddings[true_sparse_indices]
        
        # Ensure points are marked as sparse in the DataFrame
        df.loc[true_sparse_indices, 'point_type'] = 'sparse'
        
        # Subcluster the innovation gaps with more lenient parameters
        min_subcluster_size = max(2, len(true_sparse_indices) // 10)
        sparse_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_subcluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.7,  # Increased epsilon for better gap detection
            metric='euclidean'
        )
        sparse_labels = sparse_clusterer.fit_predict(sparse_points)
        
        # Collect innovation subclusters for sorting
        innovation_subclusters = []
        for label in set(sparse_labels):
            subcluster_mask = sparse_labels == label
            subcluster_patents = sparse_patents[subcluster_mask]
            subcluster_size = len(subcluster_patents)
            
            if subcluster_size >= 2:
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
            update_progress('clustering', 'processing', f'Analyzing innovation opportunity {idx + 1} of {total_subclusters} ({subcluster["size"]} patents)...')
            cluster_insights.append({
                'type': 'innovation_subcluster',
                'id': int(subcluster['label']),
                'size': subcluster['size'],
                'label': f"Innovation Gap {idx + 1}",
                'description': subcluster['description']
            })
    else:
        cluster_insights.append({
            'type': 'innovation_subcluster',
            'id': -1,
            'size': 0,
            'label': 'No Innovation Gaps',
            'description': 'No significant innovation gaps were detected in this technology space.'
        })

    update_progress('visualization', 'processing', 'Creating interactive plot...')
    
    # Create Plotly figure with clusters
    # Ensure all points are properly categorized
    unassigned_mask = df['point_type'] == 'unassigned'
    if any(unassigned_mask):
        print(f"Warning: {sum(unassigned_mask)} points remain unassigned")
        df.loc[unassigned_mask, 'point_type'] = 'cluster'  # Default unassigned to clusters
    
    # Separate points into three categories: clusters, innovation gaps, and dense noise
    cluster_mask = df['point_type'] == 'cluster'
    innovation_gaps_mask = df['point_type'] == 'sparse'
    dense_noise_mask = df['point_type'] == 'dense_noise'
    
    # Validate and debug point type assignments before visualization
    total_points = len(df)
    cluster_count = sum(cluster_mask)
    gaps_count = sum(innovation_gaps_mask)
    transitional_count = sum(dense_noise_mask)
    
    print("\nFinal Point Distribution:")
    print(f"Total points: {total_points}")
    print(f"Cluster points: {cluster_count} ({cluster_count/total_points:.1%})")
    print(f"Innovation gaps: {gaps_count} ({gaps_count/total_points:.1%})")
    print(f"Transitional areas: {transitional_count} ({transitional_count/total_points:.1%})")
    
    # Detailed debug information
    print("\nDetailed Point Type Analysis:")
    print("Point type counts from DataFrame:")
    print(df['point_type'].value_counts())
    
    print("\nInnovation Gap Details:")
    if gaps_count > 0:
        gap_points = df[innovation_gaps_mask]
        print(f"Sample of innovation gap points:")
        print(gap_points[['title', 'point_type']].head())
        print("\nCoordinates of innovation gaps:")
        print(gap_points[['x', 'y', 'z']].head())
    else:
        print("No innovation gaps detected!")
    
    # Initialize transitional area map
    transitional_area_map = {}
    for area_idx, area in enumerate(transitional_areas):
        for idx in area['indices']:
            transitional_area_map[idx] = {'number': area_idx + 1, 'size': area['size']}
    
    # Create hover text for all points
    hover_text = []
    # Create mapping for innovation gap points to their numbers
    innovation_gap_map = {}
    innovation_gap_counter = 1
    for mask, patents in df[df['point_type'] == 'sparse'].groupby(sparse_labels):
        for idx in patents.index:
            innovation_gap_map[idx] = innovation_gap_counter
        innovation_gap_counter += 1

    # Generate hover text for each point
    for idx, row in df.iterrows():
        cluster_val = row['cluster']
        if row['point_type'] == 'sparse':
            gap_number = innovation_gap_map.get(idx, 1)  # Default to 1 if not found
            point_info = f"<br><b>Region:</b> Innovation Gap {gap_number}"
        elif row['point_type'] == 'dense_noise':
            area_info = transitional_area_map.get(idx)
            if area_info:
                point_info = f"<br><b>Region:</b> Transitional Area {area_info['number']} ({area_info['size']} patents)"
            else:
                point_info = "<br><b>Region:</b> Transitional Area"
        else:
            point_info = f"<br><b>Cluster:</b> {cluster_val}"
        text = (
            f"<b>{row['title']}</b><br><br>"
            f"<b>By:</b> {row['assignee']} ({row['year']})<br>"
            f"{point_info}<br><br>"
            f"<b>Abstract:</b><br>{row['abstract']}"
        )
        hover_text.append(text)

    # Create three separate traces: clusters, innovation gaps, and dense noise points
    cluster_trace = go.Scatter3d(
        x=df[cluster_mask]['x'],
        y=df[cluster_mask]['y'],
        z=df[cluster_mask]['z'],
        mode='markers',
        marker=dict(
            size=6,  # Made even smaller for better contrast with gaps
            color=clusters[cluster_mask],
            colorscale='Viridis',
            opacity=0.5,  # More transparent to make gaps more visible
            showscale=True,
            colorbar=dict(
                title="Clusters",
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
        name='Innovation Gaps',
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    """Server-sent events endpoint for progress updates"""
    def generate():
        connection_active = True
        while connection_active:
            try:
                data = progress_queue.get(timeout=10)  # Reduced timeout for more responsive updates
                if data == 'DONE':
                    yield f"data: {json.dumps({'step': 'complete', 'status': 'done'})}\n\n"
                    connection_active = False
                else:
                    yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
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

def update_progress(step, status='processing', message=None):
    """Update progress through the progress queue"""
    try:
        # Clear any old messages to prevent queue buildup while preserving the order
        with threading.Lock():
            while not progress_queue.empty():
                progress_queue.get_nowait()
            
            # Send the new progress update
            data = {
                'step': step,
                'status': status,
                'message': message or '',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'id': str(time.time())  # Add unique ID for each update
            }
            progress_queue.put(data)
    except Exception as e:
        print(f"Error updating progress: {e}")

@app.route('/search', methods=['POST'])
def search():
    keywords = request.form.get('keywords', '')
    if not keywords:
        return jsonify({'error': 'Please enter search keywords'})
    
    print(f"\nProcessing search request for keywords: {keywords}")
    
    try:
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
        
        # Generate embeddings progress is handled in search_patents function
        
        # Start visualization processing
        update_progress('visualization', 'Creating visualization...')
        viz_data = create_3d_visualization(patents)
        if not viz_data:
            progress_queue.put('DONE')
            return jsonify({'error': 'Error creating visualization'})
        
        # Final progress update
        update_progress('complete', 'Analysis complete!')
        progress_queue.put('DONE')
        
        return jsonify({
            'visualization': viz_data['plot'],
            'insights': viz_data['insights']
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        progress_queue.put('DONE')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)