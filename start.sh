#!/bin/bash

# Augur Seller Scoring Application Startup Script

echo "🏠 Starting Augur Seller Scoring Application..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads exports

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your database credentials before continuing."
    echo "   Press Enter when ready to continue..."
    read
fi

# Start services
echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Application URLs:"
    echo "   Main App: http://localhost:8501"
    echo "   Database: localhost:5432"
    echo "   pgAdmin: http://localhost:5050 (if enabled)"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Failed to start services. Check logs with: docker-compose logs"
    exit 1
fi
