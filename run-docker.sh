#!/bin/bash

# Audio Steganography Docker Runner
echo "🎵 Audio Steganography Docker Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to build and run production
build_and_run_production() {
    echo "🏗️  Building production Docker image..."
    docker-compose build audio-steganography
    
    echo "🚀 Starting production application..."
    docker-compose up audio-steganography
    
    echo "✅ Application is running!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
}

# Function to build and run development
build_and_run_development() {
    echo "🏗️  Building development Docker image..."
    docker-compose --profile dev build audio-steganography-dev
    
    echo "🚀 Starting development application..."
    docker-compose --profile dev up audio-steganography-dev
    
    echo "✅ Development application is running!"
    echo "🌐 Frontend: http://localhost:3001"
    echo "🔧 Backend API: http://localhost:8001"
}

# Function to stop containers
stop_containers() {
    echo "🛑 Stopping containers..."
    docker-compose down
    echo "✅ Containers stopped!"
}

# Function to show logs
show_logs() {
    echo "📋 Showing logs..."
    docker-compose logs -f
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    docker system prune -f
    echo "✅ Cleanup complete!"
}

# Main menu
case "${1:-}" in
    "prod"|"production")
        build_and_run_production
        ;;
    "dev"|"development")
        build_and_run_development
        ;;
    "stop")
        stop_containers
        ;;
    "logs")
        show_logs
        ;;
    "clean"|"cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {prod|dev|stop|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  prod, production  - Build and run production version"
        echo "  dev, development  - Build and run development version (with hot reload)"
        echo "  stop              - Stop running containers"
        echo "  logs              - Show container logs"
        echo "  clean, cleanup    - Clean up Docker resources"
        echo ""
        echo "Examples:"
        echo "  $0 prod           # Run production"
        echo "  $0 dev            # Run development"
        echo "  $0 stop           # Stop containers"
        ;;
esac 