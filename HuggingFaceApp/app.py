from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
from datetime import datetime
import os
import json
import openai

load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
PATENTSVIEW_API_KEY = os.getenv('PATENTSVIEW_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not PATENTSVIEW_API_KEY:
    raise ValueError("PATENTSVIEW_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

def search_patents(keywords, num_results=15):
    """
    Search patents using PatentsView API
    """
    # PatentsView API endpoint
    api_url = "https://search.patentsview.org/api/v1/patent/"
    
    # Construct the query - search in title and abstract
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
            "size": num_results
        }
    }
    
    try:
        # Make request to PatentsView API
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": PATENTSVIEW_API_KEY
        }
        
        response = requests.post(api_url, json=query, headers=headers)
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            return []
            
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        # Get the patents from response
        if response_data.get('error'):
            print(f"API returned error: {response_data}")
            return []
            
        patents_data = response_data.get('patents', [])
        
        if not patents_data:
            print("No patents found in response")
            return []
            
        formatted_patents = []
        for patent in patents_data:
            # Format filing date
            date_str = patent.get('patent_date', '')
            filing_year = 'N/A'
            if date_str:
                try:
                    filing_year = datetime.strptime(date_str, '%Y-%m-%d').year
                except ValueError:
                    print(f"Invalid date format: {date_str}")
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

            formatted_patent = {
                'title': patent.get('patent_title', 'N/A'),
                'assignee': assignee_org,
                'filing_year': filing_year,
                'abstract': patent.get('patent_abstract', 'N/A'),
                'link': f"https://patents.google.com/patent/{patent_id}"
            }
            formatted_patents.append(formatted_patent)
            
        print(f"Formatted {len(formatted_patents)} patents")
        return formatted_patents
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return []
    except Exception as e:
        print(f"Error searching patents: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

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
    summary = generate_summary(patents)
    
    return jsonify({
        'patents': patents,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860) 