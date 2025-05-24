from flask import Flask, render_template, request, jsonify
from serpapi import GoogleSearch
from datetime import datetime

app = Flask(__name__)

def search_patents(keywords, num_results=10):
    """
    Search patents using SerpAPI Google Patents
    """
    api_key = "1c3072feed5fcd1cdc029bbb7f27a87c7e7aac5710cf05dcd5960dfeebeb8fec"
    search_params = {
        "engine": "google_patents",
        "q": keywords,
        "api_key": api_key,
        "num": num_results
    }
    
    try:
        search = GoogleSearch(search_params)
        patents = search.get_dict().get("organic_results", [])
        
        formatted_patents = []
        for patent in patents:
            # Format filing date
            date_str = patent.get('filing_date', '')
            filing_year = 'N/A'
            if date_str:
                try:
                    filing_year = datetime.strptime(date_str, '%Y-%m-%d').year
                except ValueError:
                    pass

            formatted_patent = {
                'title': patent.get('title', 'N/A'),
                'assignee': patent.get('assignee', 'N/A'),
                'filing_year': filing_year,
                'abstract': patent.get('snippet', 'N/A'),
                'link': patent.get('patent_link', '#')
            }
            formatted_patents.append(formatted_patent)
            
        return formatted_patents
    except Exception as e:
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keywords = request.form.get('keywords', '')
    if not keywords:
        return jsonify({'error': 'Please enter search keywords'})
    
    patents = search_patents(keywords)
    return jsonify({'patents': patents})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860) 