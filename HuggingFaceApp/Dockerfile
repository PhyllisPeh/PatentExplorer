FROM python:3.9

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /code/templates

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application code
COPY ./app.py /code/app.py
COPY ./templates /code/templates

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Required environment variables:
# - PATENTSVIEW_API_KEY: Your PatentsView API key
# - OPENAI_API_KEY: Your OpenAI API key
# Example: docker run -e PATENTSVIEW_API_KEY=your_key -e OPENAI_API_KEY=your_openai_key ...

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "300"] 