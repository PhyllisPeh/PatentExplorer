from serpapi import GoogleSearch
import os
from datetime import datetime

def search_patents(keywords, api_key, num_results=10):
    """
    Search patents using SerpAPI Google Patents
    """
    search_params = {
        "engine": "google_patents",
        "q": keywords,
        "api_key": api_key,
        "num": num_results
    }
    
    try:
        search = GoogleSearch(search_params)
        patents = search.get_dict().get("organic_results", [])
        
        for i, patent in enumerate(patents, 1):
            print(f"\nPatent {i}:")
            print("=" * 80)
            
            # Title
            print(f"Title: {patent.get('title', 'N/A')}")
            
            # Assignee (Company)
            assignee = patent.get('assignee', 'N/A')
            print(f"Assignee: {assignee}")
            
            # Filing Year
            try:
                date_str = patent.get('filing_date', '')
                if date_str:
                    filing_year = datetime.strptime(date_str, '%Y-%m-%d').year
                    print(f"Filing Year: {filing_year}")
                else:
                    print("Filing Year: N/A")
            except ValueError:
                print("Filing Year: N/A")
            
            # Abstract
            print("\nAbstract:")
            print(patent.get('snippet', 'N/A'))
            
            '''
            # Claims (if available in the API response)
            if 'claims' in patent:
                print("\nMain Claim:")
                print(patent['claims'])
            '''
            
            print("=" * 80)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    '''
    # Get API key from environment variable
    api_key = os.getenv('SERPAPI_KEY')
    
    if not api_key:
        print("Please set your SerpAPI key as an environment variable named 'SERPAPI_KEY'")
        return
    '''

    while True:
        # Get search keywords from user
        keywords = input("\nEnter keywords to search patents (or 'quit' to exit): ").strip()
        
        if keywords.lower() == 'quit':
            break
        
        if not keywords:
            print("Please enter valid keywords.")
            continue
        '''
        # Get number of results
        try:
            num_results = int(input("How many results do you want to see? (default: 5): ") or 5)
        except ValueError:
            num_results = 5
            print("Invalid input. Using default value of 5 results.")
        '''
        api_key = "1c3072feed5fcd1cdc029bbb7f27a87c7e7aac5710cf05dcd5960dfeebeb8fec"
        num_results = 100

        # Search patents
        search_patents(keywords, api_key, num_results)

if __name__ == "__main__":
    main()
