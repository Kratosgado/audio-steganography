# Multi-stage Dockerfile for Audio Steganography Application
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY steg-frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY steg-frontend/ ./

# Build the frontend
RUN npm run build

# Python backend stage
FROM python:3.11-slim AS backend-builder

# Set working directory for backend
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Final stage
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from backend stage
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy backend source code
COPY --from=backend-builder /app/backend ./backend

# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package.json ./frontend/package.json

# Install serve for frontend
RUN npm install -g serve

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "Starting Audio Steganography Application..."\n\
echo "Starting backend API on port 8000..."\n\
cd /app/backend && python main.py &\n\
BACKEND_PID=$!\n\
echo "Starting frontend on port 3000..."\n\
cd /app/frontend && serve -s .next -l 3000 &\n\
FRONTEND_PID=$!\n\
echo "Application started!"\n\
echo "Frontend: http://localhost:3000"\n\
echo "Backend API: http://localhost:8000"\n\
wait $BACKEND_PID $FRONTEND_PID' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 3000 8000

# Set the startup command
CMD ["/app/start.sh"] 