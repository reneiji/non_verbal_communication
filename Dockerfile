# ---------- Stage 1: Build dependencies ----------
FROM python:3.10.6-slim AS builder

WORKDIR /app

# Install system build dependencies (only needed for compiling)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install all dependencies (including parselmouth) into /install
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt praat-parselmouth

# ---------- Stage 2: Runtime image ----------
FROM python:3.10.6-slim

WORKDIR /app

# Install runtime system dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source code
COPY . .

# (Optional) EXPOSE for local testing
EXPOSE 8501

# Run Streamlit on Heroku's $PORT
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]

# build docker image locally 
# docker build -t my-streamlit-app .

# run app with docker build locally
# docker run -e PORT=1234 -p 1234:1234 my-streamlit-app

# Build image with heroku
# docker build -t registry.heroku.com/my-streamlit-app/web .

# Push and release with heroku
# docker push registry.heroku.com/my-streamlit-app/web
# heroku container:release web -a my-streamlit-app

# open app with heroku
# heroku open -a my-streamlit-app