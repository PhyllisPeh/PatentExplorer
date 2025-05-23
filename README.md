# Patent Explorer

This Python script allows you to search for patents using the Google Patents API through SerpApi. It displays patent information including titles, abstracts, claims, assignees, and filing years.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Sign up for a SerpApi account at https://serpapi.com/ and get your API key.

3. Set your SerpApi API key as an environment variable:
   - Windows (CMD):
     ```bash
     set SERPAPI_KEY=your_api_key_here
     ```
   - Windows (PowerShell):
     ```powershell
     $env:SERPAPI_KEY="your_api_key_here"
     ```

## Usage

1. Run the script:
```bash
python patentexplorer.py
```

2. Enter your search keywords when prompted.

3. Specify the number of results you want to see (default is 5).

4. The script will display the following information for each patent:
   - Title
   - Assignee (Company)
   - Filing Year
   - Abstract
   - Main Claim (if available)

5. Type 'quit' when prompted for keywords to exit the program.

## Note

- The script requires an active internet connection.
- API usage is subject to SerpApi's pricing and terms of service.
- Make sure your API key is valid and has sufficient credits.
