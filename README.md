# Patent Explorer

A sophisticated Flask-based web application for exploring and visualizing patent data, identifying innovation opportunities, and conducting competitive analysis in patent landscapes.

## Features

### 🔍 Patent Search & Discovery
- Search patents using keywords across multiple fields
- Integration with SerpAPI's Google Patents API for comprehensive patent data
- Advanced filtering and data retrieval capabilities

### 📊 Interactive Visualizations
- 3D interactive patent landscape visualizations using Plotly
- UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
- Clustering analysis using HDBSCAN and K-means algorithms
- Dynamic color-coding by patent categories and clusters

### 🧠 AI-Powered Insights
- OpenAI integration for intelligent patent analysis
- Automated identification of innovation opportunities
- Competitive landscape analysis
- Patent gap identification and technology trend analysis

### 📈 Analytics & Reporting
- Cluster-based patent grouping and analysis
- Innovation opportunity scoring
- Downloadable visualization plots (PNG, SVG, PDF formats)
- Comprehensive insights reports in PDF format

### 💾 Session Management
- Persistent visualization storage across sessions
- Flask-Session integration for secure session handling
- Automatic data backup and recovery

## Technology Stack

- **Backend**: Flask 2.2.x, Python 3.9+
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Plotly, UMAP-learn
- **Machine Learning**: HDBSCAN, K-means clustering
- **AI Integration**: OpenAI API (v0.28.1)
- **Document Generation**: ReportLab
- **Frontend**: HTML5, Tailwind CSS, jQuery
- **Deployment**: Docker, Gunicorn

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PatentExplorer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Windows
   set SERPAPI_KEY=your_serpapi_key
   set OPENAI_API_KEY=your_openai_api_key
   set FLASK_SECRET_KEY=your_secret_key

   # Linux/Mac
   export SERPAPI_KEY=your_serpapi_key
   export OPENAI_API_KEY=your_openai_api_key
   export FLASK_SECRET_KEY=your_secret_key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

   Access the application at `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t patent-explorer .
   ```

2. **Run the container**
   ```bash
   docker run -p 7860:7860 \
     -e SERPAPI_KEY=your_serpapi_key \
     -e OPENAI_API_KEY=your_openai_api_key \
     -e FLASK_SECRET_KEY=your_secret_key \
     patent-explorer
   ```

   Access the application at `http://localhost:7860`

## API Keys Required

### SerpAPI (Google Patents)
- **Purpose**: Access to Google's comprehensive patent database
- **How to get**: Register at [SerpAPI.com](https://serpapi.com/)
- **Required for**: Patent search and data retrieval through Google Patents

### OpenAI API
- **Purpose**: AI-powered patent analysis and insights generation
- **How to get**: Register at [OpenAI Platform](https://platform.openai.com/)
- **Required for**: Innovation opportunity analysis and intelligent insights

## Usage

1. **Start a Search**
   - Enter keywords related to your area of interest
   - Click "Search Patents" to begin analysis

2. **View Visualizations**
   - Explore the interactive 3D patent landscape
   - Hover over points to see patent details
   - Use zoom and rotation controls for better navigation

3. **Analyze Insights**
   - Review AI-generated innovation opportunities
   - Examine cluster analysis results
   - Identify technology gaps and trends

4. **Export Results**
   - Download visualization plots in various formats
   - Generate comprehensive PDF reports
   - Save insights for future reference

## Project Structure

```
PatentExplorer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md            # This file
└── templates/
    └── index.html       # Main web interface
```

## Configuration

### Environment Variables
- `SERPAPI_KEY`: Required for accessing Google Patents through SerpAPI
- `OPENAI_API_KEY`: Required for AI analysis features
- `FLASK_SECRET_KEY`: Session security (auto-generated if not provided)
- `DOCKER_CONTAINER`: Automatically set in Docker environment

### Session Configuration
- Session timeout: 30 minutes
- Secure session handling with filesystem storage
- Automatic visualization persistence across sessions

## Features in Detail

### Patent Search Algorithm
- Multi-field keyword matching
- Relevance scoring and ranking
- Automatic data cleaning and preprocessing
- Duplicate detection and removal

### Visualization Engine
- UMAP-based dimensionality reduction for patent feature vectors
- HDBSCAN clustering for automatic patent grouping
- Interactive 3D scatter plots with Plotly
- Color-coded clusters and categories

### AI Analysis Pipeline
- Patent abstract and claim analysis
- Technology trend identification
- Competitive landscape mapping
- Innovation gap detection
- Market opportunity scoring

## Performance Notes

- Optimized for datasets up to 10,000 patents
- Asynchronous processing with real-time progress updates
- Efficient memory management for large datasets
- Persistent storage for quick session recovery

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure both SERPAPI_KEY and OPENAI_API_KEY are set
   - Verify API keys are valid and have sufficient quota

2. **Visualization Not Loading**
   - Check browser JavaScript console for errors
   - Ensure all required dependencies are installed
   - Verify network connectivity for CDN resources

3. **Session Issues**
   - Clear browser cookies and restart the application
   - Check that the flask_session directory is writable
   - Ensure sufficient disk space for session storage

### Debug Mode
To enable debug mode for development:
```bash
export FLASK_DEBUG=1  # Linux/Mac
set FLASK_DEBUG=1     # Windows
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: This application requires active internet connectivity for API calls to SerpAPI (Google Patents) and OpenAI services. Ensure your network allows outbound HTTPS connections on ports 443.
