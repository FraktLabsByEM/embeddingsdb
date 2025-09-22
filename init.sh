#!/bin/bash

cd /app
echo "Starting MongoDB..."
mongod --fork --logpath /var/log/mongodb.log --dbpath /appdata/mongo

echo "MongoDB running"

echo "Starting Flask API..."
python3.9 app.py