# --- STAGE 1: BUILD THE VIRTUAL ENVIRONMENT ---
FROM python:3.11-slim as builder

# Set the working directory inside the container.
WORKDIR /code

# Install build dependencies if needed (e.g., packages that need compilation).
# For most pure-Python packages, this isn't necessary.
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Create the virtual environment in a well-known location.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV

# Activate the venv and install dependencies into it.
# The `PATH` is updated for this layer and subsequent commands.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- STAGE 2: CREATE THE FINAL, CLEAN IMAGE ---
# Use a minimal base image for the final production image.
FROM python:3.11-slim

# Set the working directory again.
WORKDIR /code

# Copy the virtual environment and application code from the builder stage.
COPY --from=builder /opt/venv /opt/venv
COPY ./app /code/app

# Update the PATH to use the virtual environment's Python.
ENV PATH="/opt/venv/bin:$PATH"

# Set the command to run the application.
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
