#!/bin/sh

# Docker Test Script for Audio Steganography
echo "🧪 Testing Audio Steganography Docker Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ "$1" -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Test 1: Check if Docker is running
echo "1. Checking Docker daemon..."
docker info > /dev/null 2>&1
print_status $? "Docker daemon is running"

# Test 2: Check if docker-compose is available
echo "2. Checking Docker Compose..."
docker-compose --version > /dev/null 2>&1
print_status $? "Docker Compose is available"

# Test 3: Build the image
echo "3. Building Docker image..."
docker-compose build audio-steganography > /dev/null 2>&1
print_status $? "Docker image built successfully"

# Test 4: Start the container
echo "4. Starting container..."
docker-compose up -d audio-steganography > /dev/null 2>&1
print_status $? "Container started successfully"

# Wait for services to be ready
echo "5. Waiting for services to be ready..."
sleep 10

# Test 5: Check backend health
echo "6. Testing backend health..."
curl -f http://localhost:8000/health > /dev/null 2>&1
print_status $? "Backend health check passed"

# Test 6: Check frontend accessibility
echo "7. Testing frontend accessibility..."
curl -f http://localhost:3000 > /dev/null 2>&1
print_status $? "Frontend is accessible"

# Test 7: Test API endpoint
echo "8. Testing API endpoint..."
curl -f http://localhost:8000/ > /dev/null 2>&1
print_status $? "API endpoint is working"

# Test 8: Check container logs
echo "9. Checking container logs..."
docker-compose logs audio-steganography | grep -q "Starting Audio Steganography"
print_status $? "Container logs show proper startup"

# Show container status
echo ""
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "🌐 Service URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   Health:   http://localhost:8000/health"

# Cleanup
echo ""
echo "🧹 Cleaning up test environment..."
docker-compose down > /dev/null 2>&1
print_status $? "Test cleanup completed"

echo ""
echo -e "${GREEN}🎉 All tests completed!${NC}"
echo "If all tests passed, your Docker setup is working correctly."
echo ""
echo "To run the application:"
echo "  ./run-docker.sh prod    # Production mode"
echo "  ./run-docker.sh dev     # Development mode"
