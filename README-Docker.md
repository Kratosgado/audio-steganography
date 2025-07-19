# Audio Steganography - Docker Setup

This document explains how to run the Audio Steganography application using Docker.

## 🐳 Quick Start

### Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

### Option 1: Using the Run Script (Recommended)

```bash
# Make the script executable (if not already done)
chmod +x run-docker.sh

# Run production version
./run-docker.sh prod

# Or run development version (with hot reload)
./run-docker.sh dev

# Stop containers
./run-docker.sh stop

# View logs
./run-docker.sh logs

# Clean up Docker resources
./run-docker.sh clean
```

### Option 2: Using Docker Compose Directly

```bash
# Build and run production
docker-compose up --build

# Build and run development (with hot reload)
docker-compose --profile dev up --build

# Run in background
docker-compose up -d

# Stop containers
docker-compose down
```

### Option 3: Using Docker Directly

```bash
# Build the image
docker build -t audio-steganography .

# Run the container
docker run -p 3000:3000 -p 8000:8000 audio-steganography
```

## 🌐 Accessing the Application

Once running, you can access:

- **Frontend**: http://localhost:3000 (production) or http://localhost:3001 (development)
- **Backend API**: http://localhost:8000 (production) or http://localhost:8001 (development)
- **API Documentation**: http://localhost:8000/docs (FastAPI auto-generated docs)

## 📁 Project Structure in Container

```
/app/
├── frontend/          # Next.js frontend application
│   ├── .next/        # Built frontend files
│   ├── public/       # Static assets
│   └── package.json  # Frontend dependencies
├── backend/          # Python FastAPI backend
│   ├── main.py       # Main API server
│   ├── core_modules/ # Core steganography modules
│   └── requirements.txt
└── audio_files/      # Mounted volume for audio files
```

## 🔧 Development Mode

Development mode includes:
- Hot reloading for both frontend and backend
- Volume mounting for live code changes
- Development-specific ports (3001, 8001)

```bash
./run-docker.sh dev
```

## 🏭 Production Mode

Production mode includes:
- Optimized builds
- Single container deployment
- Production ports (3000, 8000)

```bash
./run-docker.sh prod
```

## 📊 Monitoring and Logs

```bash
# View real-time logs
./run-docker.sh logs

# Or using docker-compose
docker-compose logs -f

# View logs for specific service
docker-compose logs -f audio-steganography
```

## 🧹 Cleanup

```bash
# Stop and remove containers
./run-docker.sh stop

# Full cleanup (removes images, volumes, networks)
./run-docker.sh clean

# Or manually
docker-compose down --rmi all --volumes --remove-orphans
docker system prune -f
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the ports
   lsof -i :3000
   lsof -i :8000
   
   # Kill the process or use different ports
   docker-compose up -p 3001:3000 -p 8001:8000
   ```

2. **Build fails**
   ```bash
   # Clean and rebuild
   ./run-docker.sh clean
   ./run-docker.sh prod
   ```

3. **Audio processing issues**
   ```bash
   # Check if audio libraries are installed
   docker exec -it <container_name> apt list --installed | grep libsndfile
   ```

4. **Frontend not loading**
   ```bash
   # Check if Next.js build was successful
   docker exec -it <container_name> ls -la /app/frontend/.next
   ```

### Debug Mode

```bash
# Run with debug output
docker-compose up --build --verbose

# Access container shell
docker exec -it <container_name> /bin/bash
```

## 🚀 Deployment

### Docker Hub

```bash
# Build and tag for Docker Hub
docker build -t yourusername/audio-steganography:latest .

# Push to Docker Hub
docker push yourusername/audio-steganography:latest

# Pull and run from Docker Hub
docker run -p 3000:3000 -p 8000:8000 yourusername/audio-steganography:latest
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audio-steganography
spec:
  replicas: 1
  selector:
    matchLabels:
      app: audio-steganography
  template:
    metadata:
      labels:
        app: audio-steganography
    spec:
      containers:
      - name: audio-steganography
        image: yourusername/audio-steganography:latest
        ports:
        - containerPort: 3000
        - containerPort: 8000
```

## 📝 Environment Variables

You can customize the application using environment variables:

```bash
# In docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - NODE_ENV=production
  - SAMPLE_RATE=22050
  - MAX_FILE_SIZE=10485760  # 10MB
```

## 🔒 Security Considerations

- The application runs on HTTP by default
- For production, consider adding HTTPS
- Use environment variables for sensitive configuration
- Consider using Docker secrets for API keys

## 📈 Performance

- The container uses multi-stage builds for optimal image size
- Frontend is pre-built for faster startup
- Backend uses Python 3.11 for better performance
- Audio processing libraries are optimized for the container environment 