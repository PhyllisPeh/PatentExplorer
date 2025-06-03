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
        update_progress('search', f'Fetching page {page} of patents...')
        
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
                    update_progress('embedding', f'Processing patent {total_processed} of {MAX_PATENTS}...')
                
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
    if not patents:
        return None
        
    update_progress('clustering', 'Extracting embeddings...')
    
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
    
    update_progress('clustering', 'Applying UMAP dimensionality reduction...')
    
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings_array)
    
    update_progress('clustering', 'Performing DBSCAN clustering...')
    
    # Create DataFrame for plotting
    df = pd.DataFrame(metadata)
    df['x'] = embedding_3d[:, 0]
    df['y'] = embedding_3d[:, 1]
    df['z'] = embedding_3d[:, 2]
    
    # --- Improved HDBSCAN clustering logic for sparse region detection ---
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_3d)

    n_points = len(scaled_embeddings)
    # Adjust min_cluster_size to be more conservative for smaller datasets
    min_cluster_size = max(3, min(20, int(n_points * 0.015)))  # 1.5% of data, between 3 and 20
    min_samples = max(2, min(10, int(n_points * 0.008)))      # 0.8% of data, between 2 and 10

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
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusters = hdb.fit_predict(scaled_embeddings)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print(f"\nClustering Statistics (try {retry+1}):")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of patents in sparse regions: {n_noise}")
        print(f"Total number of patents: {len(clusters)}")
        
        # If too many clusters, merge by increasing min_cluster_size
        if n_clusters > max_clusters:
            print("Too many clusters, increasing min_cluster_size...")
            min_cluster_size = int(min_cluster_size * 1.5)
            min_samples = int(min_samples * 1.2)
            retry += 1
            continue
        # If no sparse regions, retry with smaller min_samples and min_cluster_size
        if n_noise == 0:
            print("No sparse regions detected, decreasing min_samples and min_cluster_size...")
            min_cluster_size = max(2, int(min_cluster_size * 0.7))
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
    
    # --- Analyze clusters ---
    cluster_insights = []  # Initialize insights list
    
    # First analyze main clusters with new IDs
    for new_id, (_, size, cluster_patents) in enumerate(cluster_info):
        description = analyze_patent_group(cluster_patents, 'cluster', new_id)
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
    
    if len(noise_points) >= 3:
        print(f"\nStructural Analysis for Innovation Gap Detection:")
        
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
            
            # Calculate isolation scores
            density_isolation = density_ratios < 0.3  # Point density is less than 30% of nearest cluster
            spatial_isolation = cluster_distances > np.percentile(cluster_distances, 75)
            
            # Calculate structural stability
            # Check if neighboring points are also sparse
            structural_stability = np.zeros(len(noise_points), dtype=bool)
            for i, neighbors in enumerate(local_indices):
                neighbor_densities = local_densities[neighbors]
                # Point is stable if most neighbors are also sparse
                structural_stability[i] = np.mean(neighbor_densities) < np.percentile(local_densities, 25)
            
            # Combine criteria with weights
            true_sparse_indices = [
                idx for i, idx in enumerate(noise_indices)
                if (density_isolation[i] and spatial_isolation[i]) or  # Clear isolation
                   (density_isolation[i] and structural_stability[i]) or  # Stable sparse region
                   (spatial_isolation[i] and structural_stability[i])  # Stable distant region
            ]
            
            dense_noise_indices = [idx for idx in noise_indices if idx not in true_sparse_indices]
            
            # Print detailed analysis
            print("\nInnovation Gap Analysis:")
            print(f"Points with density isolation: {np.sum(density_isolation)}")
            print(f"Points with spatial isolation: {np.sum(spatial_isolation)}")
            print(f"Points with structural stability: {np.sum(structural_stability)}")
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
        
        # Mark points in DataFrame
        df['point_type'] = 'cluster'  # Default type
        df.loc[true_sparse_indices, 'point_type'] = 'sparse'
        df.loc[dense_noise_indices, 'point_type'] = 'dense_noise'
    
    # --- Handle dense noise points as transitional areas ---
    if len(dense_noise_indices) >= 3:
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
        
        # Analyze each transitional area
        for dense_label in unique_dense_labels:
            dense_mask = dense_labels == dense_label
            area_indices = [dense_noise_indices[i] for i, is_member in enumerate(dense_mask) if is_member]
            
            if len(area_indices) >= min_size:
                area_patents = df.iloc[area_indices]
                description = analyze_patent_group(area_patents, 'transitional', dense_label)
                cluster_insights.append({
                    'type': 'transitional',
                    'id': int(dense_label),
                    'size': len(area_indices),
                    'label': f"Transitional Area {dense_label + 1}",
                    'description': description
                })
        
        # Handle scattered transitional points
        scattered_mask = dense_labels == -1
        scattered_indices = [dense_noise_indices[i] for i, is_member in enumerate(scattered_mask) if is_member]
        if len(scattered_indices) > 0:
            scattered_patents = df.iloc[scattered_indices]
            description = analyze_patent_group(scattered_patents, 'transitional', -1)
            cluster_insights.append({
                'type': 'transitional',
                'id': -1,
                'size': len(scattered_indices),
                'label': "Scattered Transitional Points",
                'description': description
            })
            print(f"Found {len(scattered_indices)} scattered transitional points")

    # --- Analyze innovation gaps ---
    if len(true_sparse_indices) > 0:
        sparse_patents = df.iloc[true_sparse_indices]
        sparse_points = scaled_embeddings[true_sparse_indices]
        
        # Subcluster the innovation gaps
        min_subcluster_size = max(2, len(true_sparse_indices) // 10)
        sparse_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_subcluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.5,
            metric='euclidean'
        )
        sparse_labels = sparse_clusterer.fit_predict(sparse_points)
        
        # Analyze each innovation subcluster
        for label in set(sparse_labels):
            subcluster_mask = sparse_labels == label
            subcluster_patents = sparse_patents[subcluster_mask]
            subcluster_size = len(subcluster_patents)
            
            if subcluster_size >= 2:
                subcluster_label = "Scattered Innovation Points" if label == -1 else f"Innovation Subcluster {label + 1}"
                description = analyze_patent_group(subcluster_patents, 'innovation_subcluster', label)
                cluster_insights.append({
                    'type': 'innovation_subcluster',
                    'id': int(label),
                    'size': subcluster_size,
                    'label': subcluster_label,
                    'description': description
                })
    else:
        cluster_insights.append({
            'type': 'innovation_subcluster',
            'id': -1,
            'size': 0,
            'label': 'No Innovation Gaps',
            'description': 'No significant innovation gaps were detected in this technology space.'
        })

    update_progress('visualization', 'Creating interactive plot...')
    
    # Create Plotly figure with clusters
    # Separate points into three categories: clusters, innovation gaps, and dense noise
    cluster_mask = df['point_type'] == 'cluster'
    innovation_gaps_mask = df['point_type'] == 'sparse'
    dense_noise_mask = df['point_type'] == 'dense_noise'
    
    # Create hover text with cluster information
    hover_text = []
    for idx, row in df.iterrows():
        cluster_val = row['cluster']
        if row['point_type'] == 'sparse':
            point_info = "<br><b>Region:</b> Innovation Gap"
        elif row['point_type'] == 'dense_noise':
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
            size=10,
            color=clusters[cluster_mask],
            colorscale='Viridis',
            opacity=0.8,
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
            size=12,
            color='red',
            symbol='diamond',
            opacity=0.9,
            line=dict(
                color='white',
                width=1
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
            size=8,  # Slightly smaller than other points
            color='orange',  # Distinct color for dense noise
            symbol='circle',
            opacity=0.6,  # More transparent
            line=dict(
                color='white',
                width=1
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
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
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
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )
    
    # Add hover template configuration
    fig.update_traces(
        hovertemplate='%{text}<extra></extra>'
    )
    
    update_progress('visualization', 'Finalizing visualization...')
    
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
        while True:
            try:
                data = progress_queue.get(timeout=30)  # 30 second timeout
                if data == 'DONE':
                    break
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                break
    return Response(generate(), mimetype='text/event-stream')

def update_progress(step, status='processing'):
    """Update progress through the progress queue"""
    progress_queue.put({
        'step': step,
        'status': status,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

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
            
        # Search for patents
        update_progress('search')
        patents = search_patents(keywords)
        if not patents:
            return jsonify({'error': 'No patents found or an error occurred'})
        
        # Generate embeddings
        update_progress('embedding')
        
        # Cluster analysis
        update_progress('clustering')
        
        # Innovation analysis
        update_progress('analysis')
        
        # Create visualization
        update_progress('visualization')
        viz_data = create_3d_visualization(patents)
        if not viz_data:
            return jsonify({'error': 'Error creating visualization'})
        
        # Signal completion
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