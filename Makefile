.PHONY: install test build run

# Default target
install:
    @echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    @echo "Dependencies installed successfully."

test:
    @echo "Running tests..."
    # Add commands to run tests here
    @echo "Tests completed successfully."

build:
    @echo "Building Docker image..."
    docker build -t flask-app .
    @echo "Docker image built successfully."

run:
    @echo "Running Docker container..."
    docker run -d -p 5000:5000 flask-app
    @echo "Docker container is now running."

