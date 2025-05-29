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

load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_PATENTS = 300  # Limit number of patents to process
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
    
    # Apply DBSCAN clustering
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_3d)
    dbscan = DBSCAN(eps=0.75, min_samples=5)
    clusters = dbscan.fit_predict(scaled_embeddings)
    
    update_progress('analysis', 'Analyzing clusters and opportunities...')
    
    # Print clustering statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"\nClustering Statistics:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of patents in sparse regions: {n_noise}")
    print(f"Total number of patents: {len(clusters)}")
    
    if n_noise == 0:
        print("\nWarning: No sparse regions detected. Consider adjusting DBSCAN parameters.")
        dbscan = DBSCAN(eps=0.5, min_samples=7)
        clusters = dbscan.fit_predict(scaled_embeddings)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print(f"\nRetrying with stricter parameters:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of patents in sparse regions: {n_noise}")
    
    df['cluster'] = clusters
    
    update_progress('analysis', 'Generating cluster insights...')
    
    # Generate cluster insights
    cluster_insights = analyze_clusters(df, clusters, embedding_3d)
    
    update_progress('visualization', 'Creating interactive plot...')
    
    # Create hover text with cluster information
    hover_text = []
    for idx, row in df.iterrows():
        cluster_info = ""
        if row['cluster'] == -1:
            cluster_info = "<br><b>Region:</b> Sparse Area (Potential Innovation Zone)"
        else:
            cluster_info = f"<br><b>Cluster:</b> {row['cluster']}"
            
        text = (
            f"<b>{row['title']}</b><br><br>"
            f"<b>By:</b> {row['assignee']} ({row['year']})<br>"
            f"{cluster_info}<br><br>"
            f"<b>Abstract:</b><br>{row['abstract']}"
        )
        hover_text.append(text)
    
    # Create Plotly figure with clusters
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=10,
            color=clusters,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            colorbar=dict(
                title="Clusters<br>(-1: Opportunity Zones)",
                tickfont=dict(size=10),
                titlefont=dict(size=10)
            )
        ),
        text=hover_text,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        ),
        customdata=df['link'].tolist()
    )])
    
    # Update layout
    fig.update_layout(
        title="Patent Technology Landscape with Innovation Clusters",
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
        showlegend=False,
        template="plotly_dark",
        hoverlabel_align='left',
        hoverdistance=100,
        hovermode='closest'
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