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
from sklearn.cluster import KMeans
import hdbscan
import plotly.graph_objects as go
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
# Dynamic cluster limits based on dataset size for optimal technological granularity
def get_max_clusters(num_patents):
    """
    Calculate optimal maximum clusters based on dataset size.
    REVISED: More clusters for larger datasets to keep individual cluster sizes smaller.
    """
    if num_patents < 200:
        return min(8, num_patents // 20)   # Very small: 20-25 patents per cluster
    elif num_patents < 500:
        return min(12, num_patents // 30)  # Small datasets: 30-40 patents per cluster
    elif num_patents < 1000:
        return min(20, num_patents // 40)  # Medium datasets: 40-50 patents per cluster
    elif num_patents < 2000:
        return min(30, num_patents // 60)  # Large datasets: 60-70 patents per cluster
    else:
        return min(50, num_patents // 80)  # Very large datasets: 80-100 patents per cluster (increased from 30 max)

def get_optimal_cluster_size(num_patents):
    """Calculate optimal target cluster size range - ADJUSTED to account for noise point reassignment"""
    if num_patents < 500:
        return 25, 90  # min=25, max=90 (increased from 60 to allow room for noise points)
    elif num_patents < 1000:
        return 40, 100  # min=40, max=100 (increased from 80)  
    elif num_patents < 2000:
        return 50, 130  # min=50, max=130 (increased from 100)
    else:
        return 60, 150  # min=60, max=150 (increased from 120)

if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_embedding(text):
    """Get embedding for text using OpenAI API"""
    if not text or text.strip() == "":
        print(f"Warning: Empty text provided for embedding generation")
        return None
    
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text[:50]}...': {e}")
        return None

def get_embeddings_batch(texts, batch_size=100):
    """Get embeddings for multiple texts using OpenAI API in batches - MUCH FASTER!"""
    if not texts:
        return []
    
    # Filter out empty texts
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        print("Warning: No valid texts provided for batch embedding generation")
        return [None] * len(texts)
    
    print(f"Generating embeddings for {len(valid_texts)} texts in batches of {batch_size}...")
    all_embeddings = [None] * len(texts)  # Initialize with None for all positions
    
    # Process in batches
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i + batch_size]
        batch_indices = valid_indices[i:i + batch_size]
        
        try:
            update_progress('embedding', 'processing', f'Generating embeddings batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}...')
            
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            
            # Extract embeddings and place them in correct positions
            for j, embedding_data in enumerate(response['data']):
                original_index = batch_indices[j]
                all_embeddings[original_index] = embedding_data['embedding']
                
            print(f"✅ Generated {len(batch_texts)} embeddings in batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"❌ Error getting embeddings for batch {i//batch_size + 1}: {e}")
            # For failed batches, fall back to individual requests
            for j, text in enumerate(batch_texts):
                try:
                    individual_response = openai.Embedding.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    original_index = batch_indices[j]
                    all_embeddings[original_index] = individual_response['data'][0]['embedding']
                except Exception as individual_error:
                    print(f"❌ Failed individual embedding for text: {text[:50]}... Error: {individual_error}")
    
    successful_embeddings = sum(1 for emb in all_embeddings if emb is not None)
    print(f"📊 Batch embedding results: {successful_embeddings}/{len(texts)} successful ({successful_embeddings/len(texts)*100:.1f}%)")
    
    return all_embeddings

# Removed filtering functions - no longer needed since filtering was completely removed

def search_patents(keywords, page_size=100):
    """
    Search patents using Google Patents - OPTIMIZED for speed with batch embedding generation
    """
    all_patents = []
    page = 1
    total_processed = 0
    
    # First phase: Collect all patent data WITHOUT generating embeddings
    print("🔍 Phase 1: Collecting patent data from Google Patents API...")
    
    while len(all_patents) < MAX_PATENTS:
        update_progress('search', 'processing', f'Fetching page {page} of patents...')
        
        # SerpApi Google Patents API endpoint
        api_url = "https://serpapi.com/search"
        
        # Enhanced search parameters for better relevance
        # Use quotes for exact phrases and add title/abstract targeting
        enhanced_query = keywords
        
        # If keywords contain multiple terms, try to make the search more specific
        keyword_terms = [kw.strip() for kw in keywords.replace(',', ' ').split() if len(kw.strip()) > 2]
        if len(keyword_terms) > 1:
            # Create a more targeted query by requiring key terms to appear
            enhanced_query = f'({keywords}) AND ({" OR ".join(keyword_terms[:3])})'  # Focus on top 3 terms
        
        params = {
            "engine": "google_patents",
            "q": enhanced_query,
            "api_key": SERPAPI_API_KEY,
            "num": page_size,
            "start": (page - 1) * page_size
            # Note: Google Patents API doesn't support sort parameter
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

                # Format title and abstract - NO FILTERING, just collect everything
                title = patent.get('title', '').strip()
                abstract = patent.get('snippet', '').strip()  # SerpAPI uses 'snippet' for abstract
                combined_text = f"{title}\n{abstract}".strip()
                
                # No relevance filtering - accept all patents from search results
                total_processed += 1
                if total_processed % 50 == 0:  # Update progress every 50 patents
                    update_progress('search', 'processing', f'Collected {total_processed} patents from API...')
                
                # Store patent WITHOUT embedding (will generate in batch later)
                formatted_patent = {
                    'title': title,
                    'assignee': assignee,
                    'filing_year': filing_year,
                    'abstract': abstract,
                    'link': patent.get('patent_link', '') or patent.get('link', ''),  # SerpAPI provides patent_link or link
                    'combined_text': combined_text,  # Store for batch embedding generation
                    'embedding': None  # Will be filled in batch
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
    
    print(f"✅ Phase 1 complete: Collected {len(all_patents)} patents from API")
    
    # Second phase: Generate embeddings in batches (MUCH FASTER!)
    print("🧠 Phase 2: Generating embeddings in optimized batches...")
    if all_patents:
        # Extract all combined texts for batch processing
        combined_texts = [patent['combined_text'] for patent in all_patents]
        
        # Generate embeddings in batches - this is MUCH faster than individual calls
        batch_embeddings = get_embeddings_batch(combined_texts, batch_size=50)  # Smaller batches for reliability
        
        # Assign embeddings back to patents
        for i, patent in enumerate(all_patents):
            patent['embedding'] = batch_embeddings[i]
            # Remove the temporary combined_text field
            del patent['combined_text']
    
    # Calculate embedding statistics
    patents_with_embeddings = sum(1 for p in all_patents if p.get('embedding') is not None)
    patents_without_embeddings = len(all_patents) - patents_with_embeddings
    
    print(f"\n📊 Search Results Summary:")
    print(f"Total patents retrieved: {len(all_patents)} (no filtering applied)")
    print(f"Patents with valid embeddings: {patents_with_embeddings}")
    print(f"Patents without embeddings: {patents_without_embeddings}")
    
    if patents_without_embeddings > 0:
        embedding_success_rate = (patents_with_embeddings / len(all_patents)) * 100
        print(f"Embedding success rate: {embedding_success_rate:.1f}%")
        
    print(f"🚀 OPTIMIZED: Batch embedding generation instead of {len(all_patents)} individual API calls")
    print(f"⚡ Speed improvement: ~{len(all_patents)//50}x faster embedding generation")
    
    return all_patents

def analyze_patent_group(patents, group_type, label, max_retries=3):
    """Analyze patent clusters using ChatGPT with improved formatting and concise output"""
    # Extract key information from all patents in the group
    patent_count = len(patents)
    years_range = f"{patents['year'].min()}-{patents['year'].max()}"
    
    # Enhanced keyword extraction for better context
    all_titles = ' '.join(patents['title'].tolist())
    # Improved filtering to remove common patent language and focus on technical terms
    exclude_words = {
        'system', 'method', 'apparatus', 'device', 'process', 'technique',
        'with', 'using', 'thereof', 'based', 'related', 'improved', 'enhanced',
        'method', 'system', 'apparatus', 'device', 'comprising', 'including',
        'having', 'wherein', 'configured', 'adapted', 'operable', 'provided'
    }
    title_words = [word.lower() for word in re.findall(r'\b[A-Za-z][A-Za-z\-]+\b', all_titles)
                   if len(word) > 3 and word.lower() not in exclude_words]
    
    # Get top 6 most frequent technical terms (reduced for more focused analysis)
    title_freq = pd.Series(title_words).value_counts().head(6)
    key_terms = ', '.join(f"{word.title()}" for word in title_freq.index)  # Capitalize for better readability
    
    # Select diverse examples for better context (prefer different assignees if available)
    if patent_count > 3:
        # Try to get examples from different assignees for diversity
        unique_assignees = patents['assignee'].unique()
        example_patents = []
        used_assignees = set()
        
        for _, patent in patents.iterrows():
            if len(example_patents) >= 3:
                break
            if patent['assignee'] not in used_assignees or len(used_assignees) >= 3:
                example_patents.append(patent['title'])
                used_assignees.add(patent['assignee'])
        
        example_titles = " | ".join(example_patents[:3])
    else:
        example_titles = " | ".join(patents['title'].tolist())
    
    # Extract top assignees for competitive intelligence
    if patent_count >= 3:
        assignee_counts = patents['assignee'].value_counts().head(3)
        top_assignees = ", ".join([f"{assignee} ({count})" for assignee, count in assignee_counts.items()])
    else:
        top_assignees = ", ".join(patents['assignee'].unique())
    
    # Enhanced prompt template for cluster analysis
    base_prompt = f"""Patent cluster analysis ({patent_count} patents, {years_range}):
Key players: {top_assignees}
Core technologies: {key_terms}
Sample innovations: {example_titles}

Provide concise analysis in exactly this format:
**Technology Focus:** [What specific problem/need this cluster addresses]
**Market Applications:** [Primary commercial uses and target industries]
**Innovation Trajectory:** [How this technology is evolving and future direction]"""
            
    system_prompt = "You are a patent analyst providing strategic technology insights. Focus on commercial relevance and market opportunities."
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_prompt}
                ],
                max_tokens=200,  # Increased for more detailed structured output
                temperature=0.3   # Lowered for more consistent, focused responses
            )
            
            analysis = response.choices[0]['message']['content']
            
            # Enhanced formatting for improved readability and consistency
            # Ensure consistent markdown formatting and remove redundant text
            analysis = re.sub(r'\*\*([^*]+):\*\*\s*', r'**\1:** ', analysis)  # Standardize bold formatting
            analysis = re.sub(r'(?i)technology focus:', '**Technology Focus:**', analysis)
            analysis = re.sub(r'(?i)market applications:', '**Market Applications:**', analysis)
            analysis = re.sub(r'(?i)innovation trajectory:', '**Innovation Trajectory:**', analysis)
            
            # Clean up whitespace and formatting
            analysis = re.sub(r'\n\s*\n', '\n', analysis)  # Remove multiple blank lines
            analysis = re.sub(r'^\s+', '', analysis, flags=re.MULTILINE)  # Remove leading whitespace
            analysis = analysis.strip()
            
            # Ensure each section starts on a new line for better readability
            analysis = re.sub(r'(\*\*[^*]+:\*\*)', r'\n\1', analysis)
            analysis = analysis.strip()
            
            return analysis
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2 ** (retry_count - 1))
            else:
                return f"Analysis failed: {len(patents)} patents, {years_range}"

def create_3d_visualization(patents):
    """
    Create a 3D visualization of patent embeddings using UMAP and Plotly
    """
    # Initialize variables for tracking clusters
    df = pd.DataFrame(patents)
    
    if not patents:
        return None
        
    update_progress('clustering', 'processing', 'Extracting embeddings...')
    
    # Extract embeddings and metadata
    embeddings = []
    metadata = []
    patents_with_embeddings = 0
    patents_without_embeddings = 0
    
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
            patents_with_embeddings += 1
        else:
            patents_without_embeddings += 1
            # Log the first few patents without embeddings for debugging
            if patents_without_embeddings <= 5:
                print(f"Patent without embedding: '{patent.get('title', 'No title')[:100]}...'")
    
    # Log embedding extraction results
    total_patents = len(patents)
    print(f"\nEmbedding Extraction Summary:")
    print(f"Total patents retrieved: {total_patents}")
    print(f"Patents with valid embeddings: {patents_with_embeddings}")
    print(f"Patents without embeddings: {patents_without_embeddings}")
    
    if patents_without_embeddings > 0:
        print(f"⚠️  Warning: {patents_without_embeddings} patents ({patents_without_embeddings/total_patents*100:.1f}%) will not be plotted due to missing embeddings")
        print("This can happen due to:")
        print("1. OpenAI API errors during embedding generation")
        print("2. Empty or invalid patent text")
        print("3. Network connectivity issues")
    
    if not embeddings:
        print("❌ Error: No patents have valid embeddings to visualize")
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
    
    # Apply UMAP dimensionality reduction with better parameters for technology separation
    update_progress('clustering', 'processing', 'Applying optimized UMAP dimensionality reduction...')
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=20,        # Reduced from 30 for more local structure
        min_dist=0.05,         # Reduced from 0.1 for even tighter clusters
        spread=0.8,            # Reduced from 1.0 for better cluster separation
        random_state=42,
        metric='cosine'        # Added cosine metric for better semantic clustering
    )
    embedding_3d = reducer.fit_transform(embeddings_array)
    
    # Calculate optimal cluster parameters
    max_clusters = get_max_clusters(len(embeddings))
    min_cluster_size, max_cluster_size = get_optimal_cluster_size(len(embeddings))
    
    print(f"\n🎯 IMPROVED CLUSTERING STRATEGY:")
    print(f"Dataset size: {len(embeddings)} patents")
    print(f"Target cluster range: {min_cluster_size}-{max_cluster_size} patents per cluster")
    print(f"Maximum clusters allowed: {max_clusters}")
    
    update_progress('clustering', 'processing', f'Performing advanced multi-stage clustering...')
    
    # Create DataFrame for plotting
    df = pd.DataFrame(metadata)
    df['x'] = embedding_3d[:, 0]
    df['y'] = embedding_3d[:, 1]
    df['z'] = embedding_3d[:, 2]
    
    # --- IMPROVED MULTI-STAGE CLUSTERING ALGORITHM ---
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_3d)

    n_points = len(scaled_embeddings)
    print(f"Processing {n_points} patents with improved clustering algorithm...")
    
    # Stage 1: Initial HDBSCAN with stricter parameters
    initial_min_cluster_size = max(min_cluster_size, int(n_points * 0.020))  # Increased from 0.015 to 0.020 for stricter minimum
    initial_min_samples = max(8, int(initial_min_cluster_size * 0.6))  # Increased from 0.5 to 0.6 for stricter density
    
    print(f"Stage 1 - Initial clustering: min_cluster_size={initial_min_cluster_size}, min_samples={initial_min_samples}")
    
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=initial_min_cluster_size,
        min_samples=initial_min_samples,
        cluster_selection_epsilon=0.03,  # Reduced from 0.05 for tighter clusters
        cluster_selection_method='eom',
        metric='euclidean',
        alpha=1.2  # Increased from 1.0 for even more conservative clustering
    )
    initial_clusters = hdb.fit_predict(scaled_embeddings)
    
    # Stage 2: Subdivide oversized clusters
    print("Stage 2 - Subdividing oversized clusters...")
    final_clusters = initial_clusters.copy()
    next_cluster_id = max(initial_clusters) + 1 if len(set(initial_clusters)) > 1 else 0
    
    cluster_subdivisions = 0
    for cluster_id in set(initial_clusters):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = initial_clusters == cluster_id
        cluster_size = sum(cluster_mask)
        
        # If cluster is too large, subdivide it more aggressively
        if cluster_size > max_cluster_size:
            print(f"  Subdividing cluster {cluster_id} ({cluster_size} patents) - TOO LARGE")
            cluster_subdivisions += 1
            
            # Extract data for this oversized cluster
            cluster_data = scaled_embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Calculate how many subclusters we need - MORE AGGRESSIVE subdivision
            target_size = max_cluster_size * 0.6  # Target 60% of max size for better buffer
            n_subclusters = max(2, int(np.ceil(cluster_size / target_size)))
            # Cap at reasonable maximum but allow more splits if needed
            n_subclusters = min(12, n_subclusters)  # Increased from 10 to 12
            print(f"    Splitting into {n_subclusters} subclusters (target size: {target_size:.0f})...")
            
            # Use KMeans for controlled subdivision
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subclusters = kmeans.fit_predict(cluster_data)
            
            # Assign new cluster IDs
            for i, subcluster_id in enumerate(subclusters):
                original_idx = cluster_indices[i]
                if subcluster_id == 0:
                    # Keep first subcluster with original ID
                    final_clusters[original_idx] = cluster_id
                else:
                    # Assign new IDs to other subclusters
                    final_clusters[original_idx] = next_cluster_id + subcluster_id - 1
            
            next_cluster_id += n_subclusters - 1
    
    print(f"Subdivided {cluster_subdivisions} oversized clusters")
    
    # Stage 2.5: Additional validation and forced subdivision for any remaining oversized clusters
    print("Stage 2.5 - Final oversized cluster validation...")
    additional_subdivisions = 0
    for cluster_id in set(final_clusters):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = final_clusters == cluster_id
        cluster_size = sum(cluster_mask)
        
        # Force subdivision of any clusters still over the limit
        if cluster_size > max_cluster_size:
            print(f"  FORCING additional subdivision of cluster {cluster_id} ({cluster_size} patents)")
            additional_subdivisions += 1
            
            # Extract data for this still-oversized cluster
            cluster_data = scaled_embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Force more aggressive subdivision
            target_size = max_cluster_size * 0.5  # Even more aggressive - 50% of max
            n_subclusters = max(3, int(np.ceil(cluster_size / target_size)))
            n_subclusters = min(20, n_subclusters)  # Allow up to 20 splits if needed
            print(f"    FORCING split into {n_subclusters} subclusters...")
            
            # Use KMeans for forced subdivision
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subclusters = kmeans.fit_predict(cluster_data)
            
            # Assign new cluster IDs
            for i, subcluster_id in enumerate(subclusters):
                original_idx = cluster_indices[i]
                if subcluster_id == 0:
                    # Keep first subcluster with original ID
                    final_clusters[original_idx] = cluster_id
                else:
                    # Assign new IDs to other subclusters
                    final_clusters[original_idx] = next_cluster_id + subcluster_id - 1
            
            next_cluster_id += n_subclusters - 1
    
    if additional_subdivisions > 0:
        print(f"Performed {additional_subdivisions} additional forced subdivisions")
    else:
        print("No additional subdivisions needed - all clusters within size limits")
    
    # Stage 3: Handle noise points more intelligently with size constraints
    noise_mask = final_clusters == -1
    noise_count = sum(noise_mask)
    
    if noise_count > 0:
        print(f"Stage 3 - Reassigning {noise_count} noise points with size constraints...")
        
        # Get cluster centers and current sizes (excluding noise)
        cluster_centers = []
        cluster_labels = []
        cluster_sizes = {}
        for label in set(final_clusters):
            if label != -1:
                cluster_mask = final_clusters == label
                center = np.mean(scaled_embeddings[cluster_mask], axis=0)
                cluster_centers.append(center)
                cluster_labels.append(label)
                cluster_sizes[label] = sum(cluster_mask)
        
        if cluster_centers:
            cluster_centers = np.array(cluster_centers)
            noise_points = scaled_embeddings[noise_mask]
            
            # Find nearest clusters for each noise point
            nbrs = NearestNeighbors(n_neighbors=min(3, len(cluster_centers))).fit(cluster_centers)
            distances, nearest_indices = nbrs.kneighbors(noise_points)
            
            # Use a tighter distance threshold for reassignment
            max_distance = np.percentile(distances[:, 0], 60)  # Use 60th percentile instead of 75th
            
            noise_indices = np.where(noise_mask)[0]
            reassigned_count = 0
            rejected_too_far = 0
            rejected_too_large = 0
            
            # Calculate size buffer - leave room for some noise points
            size_buffer = max_cluster_size * 0.85  # Only allow clusters to grow to 85% of max
            
            for i, (row_distances, row_nearest_indices) in enumerate(zip(distances, nearest_indices)):
                assigned = False
                
                # Try each of the nearest clusters in order
                for dist, nearest_idx in zip(row_distances, row_nearest_indices):
                    if dist > max_distance:
                        break  # All remaining will be too far
                    
                    target_label = cluster_labels[nearest_idx]
                    current_size = cluster_sizes[target_label]
                    
                    # Only assign if cluster has room to grow
                    if current_size < size_buffer:
                        final_clusters[noise_indices[i]] = target_label
                        cluster_sizes[target_label] += 1  # Update size tracker
                        reassigned_count += 1
                        assigned = True
                        break
                    else:
                        rejected_too_large += 1
                
                if not assigned and row_distances[0] <= max_distance:
                    rejected_too_far += 1
            
            print(f"  Reassigned {reassigned_count}/{noise_count} noise points to nearby clusters")
            print(f"  Rejected {rejected_too_large} points (target clusters too large)")
            print(f"  Rejected {rejected_too_far} points (too far from suitable clusters)")
            remaining_noise = noise_count - reassigned_count
            if remaining_noise > 0:
                print(f"  {remaining_noise} points remain as noise to prevent oversized clusters")
    
    # Stage 4: Final post-noise cleanup - subdivide any clusters that grew too large
    print("Stage 4 - Post-noise subdivision check...")
    final_subdivisions = 0
    for cluster_id in set(final_clusters):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = final_clusters == cluster_id
        cluster_size = sum(cluster_mask)
        
        # If cluster grew too large after noise reassignment, subdivide again
        if cluster_size > max_cluster_size:
            print(f"  Post-noise subdivision of cluster {cluster_id} ({cluster_size} patents)")
            final_subdivisions += 1
            
            # Extract data for this oversized cluster
            cluster_data = scaled_embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Very aggressive subdivision for final cleanup
            target_size = max_cluster_size * 0.7  # Target 70% of max size
            n_subclusters = max(2, int(np.ceil(cluster_size / target_size)))
            n_subclusters = min(8, n_subclusters)  # Reasonable cap
            print(f"    Final split into {n_subclusters} subclusters...")
            
            # Use KMeans for final subdivision
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subclusters = kmeans.fit_predict(cluster_data)
            
            # Assign new cluster IDs
            for i, subcluster_id in enumerate(subclusters):
                original_idx = cluster_indices[i]
                if subcluster_id == 0:
                    # Keep first subcluster with original ID
                    final_clusters[original_idx] = cluster_id
                else:
                    # Assign new IDs to other subclusters
                    final_clusters[original_idx] = next_cluster_id + subcluster_id - 1
            
            next_cluster_id += n_subclusters - 1
    
    if final_subdivisions > 0:
        print(f"Performed {final_subdivisions} final post-noise subdivisions")
    else:
        print("No post-noise subdivisions needed")
    
    clusters = final_clusters

    df['cluster'] = clusters

    # --- Gather clusters and analyze them ---
    cluster_info = []
    n_clusters = len(set(clusters))
    
    for label in set(clusters):
        cluster_mask = clusters == label
        cluster_patents = df[cluster_mask]
        if len(cluster_patents) > 0:
            cluster_info.append((label, len(cluster_patents), cluster_patents))
    
    # Sort clusters by size in descending order
    cluster_info.sort(key=lambda x: x[1], reverse=True)
    
    # Limit the number of clusters to calculated maximum
    if len(cluster_info) > max_clusters:
        print(f"\nLimiting clusters from {len(cluster_info)} to {max_clusters} largest clusters")
        
        # Keep only the top max_clusters largest clusters
        main_clusters = cluster_info[:max_clusters]
        small_clusters = cluster_info[max_clusters:]
        
        # Reassign patents from small clusters to the nearest large cluster
        if small_clusters:
            print(f"Reassigning {len(small_clusters)} smaller clusters to larger ones...")
            
            # Get embeddings for main cluster centers
            main_cluster_centers = []
            main_cluster_labels = []
            for old_label, size, cluster_patents in main_clusters:
                cluster_mask = clusters == old_label
                center = np.mean(scaled_embeddings[cluster_mask], axis=0)
                main_cluster_centers.append(center)
                main_cluster_labels.append(old_label)
            
            main_cluster_centers = np.array(main_cluster_centers)
            
            # Reassign each small cluster to nearest main cluster
            for small_label, small_size, _ in small_clusters:
                small_cluster_mask = clusters == small_label
                small_cluster_center = np.mean(scaled_embeddings[small_cluster_mask], axis=0)
                
                # Find nearest main cluster
                distances = np.linalg.norm(main_cluster_centers - small_cluster_center, axis=1)
                nearest_main_idx = np.argmin(distances)
                nearest_main_label = main_cluster_labels[nearest_main_idx]
                
                # Reassign all patents in small cluster to nearest main cluster
                clusters[small_cluster_mask] = nearest_main_label
                print(f"  Merged cluster of {small_size} patents into larger cluster")
        
        # Update cluster_info to only include main clusters
        cluster_info = main_clusters
        
    # Final cluster validation and reporting
    final_cluster_info = []
    noise_count = sum(1 for c in clusters if c == -1)
    
    for label in set(clusters):
        if label != -1:  # Skip noise
            cluster_mask = clusters == label
            cluster_patents = df[cluster_mask]
            if len(cluster_patents) > 0:
                final_cluster_info.append((label, len(cluster_patents), cluster_patents))
    
    # Sort clusters by size in descending order
    final_cluster_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ FINAL CLUSTERING RESULTS:")
    print(f"Total patents processed: {len(df)}")
    print(f"Number of technology clusters: {len(final_cluster_info)}")
    print(f"Noise points (unassigned): {noise_count}")
    
    if final_cluster_info:
        sizes = [size for _, size, _ in final_cluster_info]
        avg_size = np.mean(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        print(f"Cluster size stats: min={min_size}, avg={avg_size:.1f}, max={max_size}")
        print(f"Target range was: {min_cluster_size}-{max_cluster_size} patents per cluster")
        
        # Check if we successfully avoided mega-clusters
        oversized_clusters = [size for size in sizes if size > max_cluster_size]
        if oversized_clusters:
            print(f"⚠️  WARNING: {len(oversized_clusters)} clusters STILL oversized: {oversized_clusters}")
            print(f"❌ FAILED to contain all clusters within target range!")
            
            # Log the oversized clusters for debugging
            for i, (label, size, _) in enumerate(final_cluster_info):
                if size > max_cluster_size:
                    print(f"   Oversized Cluster {i + 1}: {size} patents (EXCEEDS LIMIT of {max_cluster_size})")
        else:
            print(f"✅ SUCCESS: All clusters within target size range!")
        
        print("\nCluster Size Distribution:")
        for i, (label, size, _) in enumerate(final_cluster_info):
            if size > max_cluster_size:
                status = "❌ OVERSIZED"
                severity = f"(+{size - max_cluster_size} over limit)"
            elif min_cluster_size <= size <= max_cluster_size:
                status = "✅ OPTIMAL"
                severity = ""
            else:
                status = "⚠️  SMALL"
                severity = f"({min_cluster_size - size} under target)"
            print(f"  {status} Cluster {i + 1}: {size} patents {severity}")
    
    cluster_info = final_cluster_info
    
    # Create mapping for new cluster IDs (1-based)
    cluster_id_map = {old_label: i + 1 for i, (old_label, _, _) in enumerate(cluster_info)}
    
    # Update cluster IDs in DataFrame to be 1-based
    new_clusters = clusters.copy()
    for old_label, new_label in cluster_id_map.items():
        new_clusters[clusters == old_label] = new_label
    df['cluster'] = new_clusters
    
    update_progress('clustering', 'processing', 'Analyzing technological clusters...')
    
    # Analyze each cluster
    cluster_insights = []
    total_clusters = len(cluster_info)
    for i, (_, size, cluster_patents) in enumerate(cluster_info):
        cluster_id = i + 1  # 1-based cluster ID
        update_progress('clustering', 'processing', f'Analyzing cluster {cluster_id} of {total_clusters} ({size} patents)...')
        description = analyze_patent_group(cluster_patents, 'cluster', cluster_id)
        cluster_insights.append({
            'type': 'cluster',
            'id': cluster_id,
            'size': size,
            'label': f"Cluster {cluster_id}",
            'description': description
        })

    update_progress('visualization', 'processing', 'Creating interactive plot...')
    
    
    # Create Plotly figure with clusters only
    # Create hover text for all points
    hover_text = []
    for idx, row in df.iterrows():
        text = (
            f"<b>{row['title']}</b><br><br>"
            f"<b>By:</b> {row['assignee']} ({row['year']})<br>"
            f"<b>Cluster:</b> {int(row['cluster'])}<br><br>"
            f"<b>Abstract:</b><br>{row['abstract']}"
        )
        hover_text.append(text)

    # Create single trace for all clusters
    cluster_trace = go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['cluster'],
            colorscale='Viridis',
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title="Technology Clusters",
                tickmode="linear",
                tick0=1,
                dtick=1,
                tickfont=dict(size=10),
                titlefont=dict(size=12)
            )
        ),
        text=hover_text,
        hoverinfo='text',
        name='Technology Clusters',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        ),
        customdata=df['link'].tolist()
    )

    fig = go.Figure(data=[cluster_trace])
    
    # Update layout
    fig.update_layout(
        title="Patent Technology Landscape - Cluster Analysis",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.8, y=1.8, z=1.8)
            ),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,  # Single trace doesn't need legend
        template="plotly_dark",
        hoverlabel_align='left',
        hoverdistance=100,
        hovermode='closest'
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
        area_pattern = r'(?:Cluster)\s+(\d+)'
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
    Analyze technology clusters to identify potential innovation opportunities.
    Returns focused analysis of high-value innovation opportunities within and between technology clusters.
    """
    # Extract cluster numbers and validate
    cluster_nums = set()
    
    # Parse and validate cluster numbers with explicit error checking
    for insight in cluster_insights:
        area_type = insight.get('type', '')
        area_id = insight.get('id', -1)
        
        if area_type == 'cluster' and area_id > 0:
            cluster_nums.add(area_id)

    # Only generate analysis if we have clusters to analyze
    if not cluster_nums:
        return "No technology clusters found. Try broadening search terms or increasing patent count."

    # Create descriptions list with cluster information
    descriptions = []
    cluster_details = {}
    
    for insight in cluster_insights:
        if insight.get('description') and insight.get('type') == 'cluster':
            area_id = int(insight.get('id', -1))  # 1-based IDs
            area_size = insight.get('size', 0)
            
            desc = f"C{area_id}:{insight['description']}"
            descriptions.append(desc)
            cluster_details[area_id] = {'description': insight['description'], 'size': area_size}
    
    # Format descriptions as a string with newlines
    descriptions_text = '\n'.join(descriptions)

    prompt = f"""Technology Clusters Available:
Clusters: {', '.join(f'Cluster {n}' for n in sorted(cluster_nums))}

Cluster Descriptions:
{descriptions_text}

I need you to identify 3-4 high-value innovation opportunities in this patent technology landscape. Focus on creating REAL business value through either:
A) Cross-pollinating technologies between different clusters, OR
B) Identifying innovation gaps within individual clusters

For each opportunity:
1. Select either ONE cluster with internal innovation potential OR two complementary clusters that can be combined
2. Identify a specific technical or market gap within or between the selected clusters
3. Propose a concrete solution that addresses this gap
4. Quantify potential business impact and competitive advantage

Follow this precise format:
Opportunity N: [Title that describes the innovation]
Source: [Single cluster (e.g., "Cluster 2") OR combination (e.g., "Cluster 1 + Cluster 3")]
- Gap: [Specific technical or market gap that represents an unmet need]
- Solution: [Practical, implementable technical approach]
- Impact: [Specific business value creation - market size, efficiency gains, cost reduction]
- Timeline: [Short-term (1-2 years) or medium-term (3-5 years)]

Prioritize opportunities based on:
1. Commercial potential (market size, growth potential)
2. Technical feasibility (can be implemented with current or near-term technology)
3. Competitive advantage (uniqueness, barriers to entry)
4. Alignment with industry trends (sustainability, automation, digitalization)

Focus on practical innovations that could realistically be implemented by a company rather than theoretical or speculative concepts."""

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

# ...existing code...
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
                # Write the HTML content with click functionality
                html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Patent Technology Landscape</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: white;
        }
        #plot {
            width: 100%%;
            height: 80vh;
        }
        .info {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #2d2d2d;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="info">
        <h1>Patent Technology Landscape</h1>
        <p><strong>Instructions:</strong> Click on any point to open the corresponding Google Patents page in a new tab.</p>
        <p><strong>Legend:</strong> 
            <span style="color: #636EFA;">● Technology Clusters</span>
        </p>
    </div>
    <div id="plot"></div>
    <script>
        var plotData = %s;
        
        // Create the plot
        Plotly.newPlot('plot', plotData.data, plotData.layout);
        
        // Add click event listener
        document.getElementById('plot').on('plotly_click', function(data) {
            console.log('Plot clicked:', data);
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                let patentUrl = null;
                
                // Check for patent URL in customdata
                if (point.customdata) {
                    patentUrl = point.customdata;
                } else if (point.text && point.text.includes('US')) {
                    // Extract patent number from text and create Google Patents URL
                    const patentMatch = point.text.match(/US[\\d,]+/);
                    if (patentMatch) {
                        const patentNumber = patentMatch[0].replace(/,/g, '');
                        patentUrl = `https://patents.google.com/patent/${patentNumber}`;
                    }
                }
                
                if (patentUrl) {
                    console.log('Opening patent URL:', patentUrl);
                    window.open(patentUrl, '_blank');
                } else {
                    console.log('No patent URL found for clicked point');
                    alert('No patent link available for this point.');
                }
            }
        });
        
        // Update cursor style on hover
        document.getElementById('plot').style.cursor = 'pointer';
        
        // Add hover effect
        document.getElementById('plot').on('plotly_hover', function(data) {
            document.getElementById('plot').style.cursor = 'pointer';
        });
        
        document.getElementById('plot').on('plotly_unhover', function(data) {
            document.getElementById('plot').style.cursor = 'default';
        });
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
                    
                    # Process sections like Source: [Area], Gap, Solution, Impact
                    opp_lines = opp_text.split('\n')
                    for j, line in enumerate(opp_lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Format the first line (Source area specification) specially
                        if j == 0 and line.startswith('Source:'):
                            story.append(Paragraph(line, opportunity_style))
                        # Format any other non-bullet first line
                        elif j == 0:
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