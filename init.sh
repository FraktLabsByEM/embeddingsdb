#!/bin/bash

# echo "🚀 Starting MongoDB..."
mongod --fork --logpath /var/log/mongodb.log --dbpath /appdata/mongo

echo "✅ MongoDB running"

# echo "🚀 Starting Flask API..."
cd /app
python3.9 app.py

tail -f /dev/null