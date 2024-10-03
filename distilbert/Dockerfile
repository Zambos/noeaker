FROM python:3.9

# Set working directory
WORKDIR /app/distilbert

# Install required packages
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the FastAPI port
EXPOSE 8001

# Run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
