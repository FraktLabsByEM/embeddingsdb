#!/bin/bash

mongod --fork --logpath /var/log/mongodb.log --dbpath /appdata/mongo
echo "MongoDB running"

cd /app
python app.py

tail -f /dev/null