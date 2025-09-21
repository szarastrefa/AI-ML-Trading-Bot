#!/bin/bash

# AI/ML Trading Bot - Startup Script

echo "Starting AI/ML Trading Bot..."

# Wait for database
echo "Waiting for database connection..."
while ! nc -z postgres 5432; do
  sleep 0.1
done
echo "Database connected!"

# Wait for Redis
echo "Waiting for Redis connection..."
while ! nc -z redis 6379; do
  sleep 0.1
done
echo "Redis connected!"

# Initialize database
echo "Initializing database..."
python scripts/init_db.py

# Start the application
echo "Starting FastAPI application..."
python app/main.py