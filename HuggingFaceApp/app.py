from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
from datetime import datetime
import os
import json
import openai
import numpy as np
import pickle
from pathlib import Path

load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
PATENTSVIEW_API_KEY = os.getenv('PATENTSVIEW_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_PATENTS = 300  # Limit number of patents to process
CACHE_FILE = 'patent_embeddings_cache.pkl'

if not PATENTSVIEW_API_KEY:
    raise ValueError("PATENTSVIEW_API_KEY environment variable is not set")
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
    Search patents using PatentsView API with pagination and generate embeddings
    """
    # Load existing cache
    embedding_cache = load_cache()
    
    # PatentsView API endpoint
    api_url = "https://search.patentsview.org/api/v1/patent/"
    
    all_patents = []
    page = 1
    
    while len(all_patents) < MAX_PATENTS:
        query = {
            "q": {
                "_or": [
                    {"_text_phrase": {"patent_title": keywords}},
                    {"_text_phrase": {"patent_abstract": keywords}}
                ]
            },
            "f": [
                "patent_title",
                "patent_abstract",
                "patent_date",
                "patent_id",
                "assignees"
            ],
            "o": {
                "page": page,
                "size": page_size
            }
        }
    
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": PATENTSVIEW_API_KEY
            }
            
            response = requests.post(api_url, json=query, headers=headers)
            response_data = response.json()
            
            if response_data.get('error'):
                print(f"API returned error: {response_data}")
                break
                
            patents_data = response_data.get('patents', [])
            
            if not patents_data:
                print(f"No more patents found on page {page}")
                break
                
            for patent in patents_data:
                if len(all_patents) >= MAX_PATENTS:
                    break
                    
                # Format filing date
                date_str = patent.get('patent_date', '')
                filing_year = 'N/A'
                if date_str:
                    try:
                        filing_year = datetime.strptime(date_str, '%Y-%m-%d').year
                    except ValueError:
                        pass

                # Get first assignee organization if available
                assignee_org = 'N/A'
                assignees = patent.get('assignees', [])
                if assignees and len(assignees) > 0:
                    assignee_org = assignees[0].get('assignee_organization', 'N/A')

                # Format patent ID for Google Patents URL
                patent_id = patent.get('patent_id', '')
                if patent_id and not patent_id.startswith('US'):
                    patent_id = f"US{patent_id}"

                # Combine title and abstract for embedding
                title = patent.get('patent_title', '').strip()
                abstract = patent.get('patent_abstract', '').strip()
                combined_text = f"{title}\n{abstract}".strip()

                # Get embedding for combined text
                embedding = get_embedding(combined_text, embedding_cache)

                formatted_patent = {
                    'title': title,
                    'assignee': assignee_org,
                    'filing_year': filing_year,
                    'abstract': abstract,
                    'link': f"https://patents.google.com/patent/{patent_id}",
                    'embedding': embedding
                }
                all_patents.append(formatted_patent)
            
            print(f"Retrieved {len(patents_data)} patents from page {page}")
            
            if len(patents_data) < page_size:
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keywords = request.form.get('keywords', '')
    if not keywords:
        return jsonify({'error': 'Please enter search keywords'})
    
    print(f"\nProcessing search request for keywords: {keywords}")
    patents = search_patents(keywords)
    if not patents:
        return jsonify({'error': 'No patents found or an error occurred'})
    
    # Generate summary using ChatGPT
    # summary = generate_summary(patents)
    return jsonify({
        'patents': patents,
        'summary': None  # Set to None since we're not generating summaries currently
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860) 