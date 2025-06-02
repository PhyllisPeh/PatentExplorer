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
MAX_PATENTS = 1000  # Limit number of patents to process
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

def generate_summary(patents):
    """
    Generate a summary of the patents using ChatGPT
    """
    if not patents:
        return "No patents to summarize."
    
    # Prepare the prompt with patent information
    prompt = "Please provide a concise summary of these patents:\n\n"
    for patent in patents[:5]:  # Limit to first 5 patents to stay within token limits
        prompt += f"Title: {patent['title']}\n"
        prompt += f"Abstract: {patent['abstract']}\n"
        prompt += f"Assignee: {patent['assignee']}\n"
        prompt += f"Year: {patent['filing_year']}\n\n"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a patent expert. Provide a clear and concise summary of the following patents, highlighting key innovations and common themes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        print("Finish reason:", response.choices[0].finish_reason)
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Error generating summary."

def analyze_clusters(df, labels, embeddings_3d):
    """
    Generate descriptions for patent clusters and identify opportunity zones
    """
    unique_labels = np.unique(labels)
    cluster_insights = []
    
    # Analyze each cluster (including noise points labeled as -1)
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_patents = df[cluster_mask]
        cluster_points = embeddings_3d[cluster_mask]
        
        if label == -1:
            # Analyze sparse regions (potential opportunity zones)
            if len(cluster_patents) > 0:
                titles = "\n".join(cluster_patents['title'].tolist())
                assignees = ", ".join(cluster_patents['assignee'].unique())
                years = f"{cluster_patents['year'].min()} - {cluster_patents['year'].max()}"
                
                prompt = f"""Analyze these {len(cluster_patents)} patents that are in sparse regions of the technology landscape:

Patents:
{titles}

Key assignees: {assignees}
Years: {years}

Please provide:
1. A brief description of these isolated technologies
2. Potential innovation opportunities in this space
3. Why these areas might be underexplored
Keep the response concise (max 3 sentences per point)."""

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a patent and technology expert analyzing innovation opportunities."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    cluster_insights.append({
                        'type': 'opportunity_zone',
                        'size': len(cluster_patents),
                        'description': response['choices'][0]['message']['content']
                    })
                except Exception as e:
                    print(f"Error generating opportunity zone analysis: {e}")
        else:
            # Analyze regular clusters
            if len(cluster_patents) > 0:
                titles = "\n".join(cluster_patents['title'].tolist())
                assignees = ", ".join(cluster_patents['assignee'].unique())
                years = f"{cluster_patents['year'].min()} - {cluster_patents['year'].max()}"
                
                prompt = f"""Analyze this cluster of {len(cluster_patents)} related patents:

Patents:
{titles}

Key assignees: {assignees}
Years: {years}

Please provide a concise (2-3 sentences) summary of:
1. The main technology focus of this cluster
2. Current development status and trends"""

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a patent and technology expert analyzing innovation clusters."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    cluster_insights.append({
                        'type': 'cluster',
                        'id': int(label),
                        'size': len(cluster_patents),
                        'description': response['choices'][0]['message']['content']
                    })
                except Exception as e:
                    print(f"Error generating cluster analysis: {e}")
    
    return cluster_insights

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

    # --- Further cluster the noise points to find multiple sparse regions ---
    noise_mask = df['cluster'] == -1
    noise_points = scaled_embeddings[noise_mask]
    noise_indices = df[noise_mask].index
    noise_labels = None
    n_sparse_regions = 0

    # Improved: Use HDBSCAN on noise points to find dense subclusters, and only treat truly scattered points as sparse regions
    sparse_region_labels = {}
    promoted_cluster_labels = {}  # New dictionary for promoted cluster labels
    if len(noise_points) > 0:
        # Calculate remaining cluster budget and target noise
        current_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        remaining_cluster_budget = max_clusters - current_clusters
        
        # First, analyze the density and relationships of noise points
        n_neighbors = min(5, len(noise_points) - 1)  # Use 5 neighbors or all points minus one
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(noise_points)
        distances, _ = nbrs.kneighbors(noise_points)
        avg_distances = np.mean(distances, axis=1)
        
        # Calculate density threshold using the mean distance to neighbors
        mean_distance = np.mean(avg_distances)
        std_distance = np.std(avg_distances)
        density_threshold = mean_distance + std_distance
        
        # Identify true sparse points (points with high average distance to neighbors)
        sparse_mask = avg_distances > density_threshold
        true_sparse_indices = [idx for i, idx in enumerate(noise_indices) if sparse_mask[i]]
        dense_noise_indices = [idx for i, idx in enumerate(noise_indices) if not sparse_mask[i]]
        
        print(f"\nDensity Analysis of Noise Points:")
        print(f"Total noise points: {len(noise_points)}")
        print(f"Average distance between points: {mean_distance:.3f}")
        print(f"Density threshold: {density_threshold:.3f}")
        print(f"True sparse points: {len(true_sparse_indices)}")
        print(f"Dense noise points: {len(dense_noise_indices)}")
        
        # If we have dense noise points, try to form subclusters
        if len(dense_noise_indices) > 0:
            dense_noise_points = scaled_embeddings[dense_noise_indices]
            
            # Use more aggressive clustering for dense noise points
            subcluster_min_size = max(3, len(dense_noise_points) // 10)  # At least 10% of dense noise points
            hdb_noise = hdbscan.HDBSCAN(
                min_cluster_size=subcluster_min_size,
                min_samples=max(3, subcluster_min_size // 2),
                cluster_selection_epsilon=0.3,
                cluster_selection_method='leaf'
            )
            dense_noise_labels = hdb_noise.fit_predict(dense_noise_points)
            
            # Promote dense subclusters
            next_cluster_id = df['cluster'].max() + 1
            dense_noise_label_map = {}
            promoted_clusters = 0
            
            for i, idx in enumerate(dense_noise_indices):
                if dense_noise_labels[i] != -1 and promoted_clusters < remaining_cluster_budget:
                    dense_label = int(dense_noise_labels[i])
                    if dense_label not in dense_noise_label_map:
                        if promoted_clusters >= remaining_cluster_budget:
                            continue
                        dense_noise_label_map[dense_label] = next_cluster_id + len(dense_noise_label_map)
                        promoted_clusters += 1
                    new_cluster_id = dense_noise_label_map[dense_label]
                    df.at[idx, 'cluster'] = new_cluster_id
                    promoted_cluster_labels[new_cluster_id] = f"Cluster {new_cluster_id}"
        
        # Mark true sparse points
        for idx in true_sparse_indices:
            df.at[idx, 'cluster'] = -1
            
        # Update clusters array to match DataFrame for visualization
        clusters = df['cluster'].values
        
        # Count final sparse points
        truly_sparse_count = len(true_sparse_indices)
        
        print(f"\nFinal Clustering Results:")
        print(f"Promoted {promoted_clusters if 'promoted_clusters' in locals() else 0} dense subclusters to clusters")
        print(f"Identified {truly_sparse_count} true innovation gaps")
        print(f"Total clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
        
        # If we found no true sparse regions, add a note about it
        if truly_sparse_count == 0:
            print("\nNote: No significant innovation gaps detected in this technology space.")
            print("This suggests a well-developed technology area with good patent coverage.")
        
        # Update n_noise to reflect only truly sparse points
        n_noise = truly_sparse_count

    update_progress('analysis', 'Generating cluster insights...')

    # --- Limit AI cost: summarize only largest clusters and sparse regions ---
    cluster_sizes = df[df['cluster'] != -1]['cluster'].value_counts()
    top_clusters = cluster_sizes.nlargest(max_clusters).index.tolist()
    df['summarize'] = df['cluster'].apply(lambda x: int(x) in [int(c) for c in top_clusters] or x not in top_clusters)

    def analyze_clusters_limited(df, labels, embeddings_3d):
        unique_labels = np.unique(labels)
        cluster_insights = []

        # Gather all clusters and their sizes (excluding -1 and sparse regions for now)
        cluster_info_list = []
        for label in unique_labels:
            if label == -1 or (label in sparse_region_labels):
                continue
            cluster_mask = labels == label
            cluster_patents = df[cluster_mask]
            if len(cluster_patents) > 0:
                cluster_info_list.append((label, len(cluster_patents), cluster_patents))

        # Sort clusters by size descending
        cluster_info_list.sort(key=lambda x: x[1], reverse=True)

        # Add sorted clusters to insights
        for label, size, cluster_patents in cluster_info_list:
            titles = "; ".join(cluster_patents['title'].tolist()[:2])
            assignees = ", ".join(cluster_patents['assignee'].unique())
            years = f"{cluster_patents['year'].min()} - {cluster_patents['year'].max()}"
            prompt = f"""Cluster {label}

Titles: {titles}
Assignees: {assignees}
Years: {years}

- Main: What is the main technology focus? (1-2 short sentences)
- Trend: What is the current trend? (1-2 short sentences)
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a patent expert. For each cluster, answer with two clear, short sentences: one for the main technology, one for the trend."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=80,
                    temperature=0.4
                )
                cluster_insights.append({
                    'type': 'cluster',
                    'id': int(label),
                    'size': size,
                    'label': f"Cluster {label}",
                    'description': response['choices'][0]['message']['content']
                })
            except Exception as e:
                print(f"Error generating cluster analysis: {e}")

        # Now handle sparse regions (label == -1 and subclusters)
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_patents = df[cluster_mask]
            is_sparse_region = False
            sparse_label = None
            if label == -1:
                if len(cluster_patents) == 0:
                    continue
                is_sparse_region = True
                sparse_label = "Sparse Region (Innovation Gap)"
            elif label in sparse_region_labels:
                is_sparse_region = True
                sparse_label = f"{sparse_region_labels[label]} (Innovation Gap)"

            if is_sparse_region and len(cluster_patents) > 0:
                titles = "; ".join(cluster_patents['title'].tolist()[:2])
                assignees = ", ".join(cluster_patents['assignee'].unique())
                years = f"{cluster_patents['year'].min()} - {cluster_patents['year'].max()}"
                prompt = f"""{sparse_label}

Titles: {titles}
Assignees: {assignees}
Years: {years}

- Main: What is the main technology here? (1-2 short sentences)
- Opportunity: What is a possible innovation opportunity? (1-2 short sentences)
"""
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a patent expert. For each region, answer with two clear, short sentences: one for the main technology, one for the opportunity. Use the region label as the heading."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=80,
                        temperature=0.4
                    )
                    cluster_insights.append({
                        'type': 'opportunity_zone',
                        'id': int(label),
                        'size': len(cluster_patents),
                        'label': sparse_label,
                        'description': response['choices'][0]['message']['content']
                    })
                except Exception as e:
                    print(f"Error generating opportunity zone analysis: {e}")

        if not any((l == -1 or (df[df['cluster'] == l].index.isin(noise_indices).any())) for l in unique_labels):
            cluster_insights.append({
                'type': 'opportunity_zone',
                'size': 0,
                'label': "No Sparse Regions",
                'description': "No sparse regions (innovation gaps) were detected in this search."
            })
        return cluster_insights

    # Use the limited cluster analysis
    cluster_insights = analyze_clusters_limited(df, df['cluster'].values, embedding_3d)

    update_progress('visualization', 'Creating interactive plot...')
    
    # Create hover text with cluster information
    hover_text = []
    for idx, row in df.iterrows():
        cluster_val = row['cluster']
        if cluster_val == -1:
            cluster_info = "<br><b>Region:</b> Sparse Region (Innovation Gap)"
        elif cluster_val in promoted_cluster_labels:
            cluster_info = f"<br><b>Cluster:</b> {promoted_cluster_labels[cluster_val]}"
        else:
            cluster_info = f"<br><b>Cluster:</b> {cluster_val}"
        text = (
            f"<b>{row['title']}</b><br><br>"
            f"<b>By:</b> {row['assignee']} ({row['year']})<br>"
            f"{cluster_info}<br><br>"
            f"<b>Abstract:</b><br>{row['abstract']}"
        )
        hover_text.append(text)
    
    # Create Plotly figure with clusters
    # Separate innovation gaps from clusters for different coloring
    innovation_gaps_mask = clusters == -1
    cluster_mask = ~innovation_gaps_mask

    # Create two separate traces: one for clusters, one for innovation gaps
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
            size=12,  # Slightly larger to highlight gaps
            color='red',  # Distinct color for innovation gaps
            symbol='diamond',  # Different symbol for innovation gaps
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

    fig = go.Figure(data=[cluster_trace, innovation_gaps_trace])
    
    # Update layout
    fig.update_layout(
        title="Patent Technology Landscape with Innovation Gaps",
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
        showlegend=True,  # Show legend to distinguish clusters from innovation gaps
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