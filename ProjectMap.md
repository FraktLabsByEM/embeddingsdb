/embedding-service
│── data/               # Data preserve
|   │── mongo           # Mongo data
|   └── faiss           # Faiss indexes
│── python/             # Python scripts
│── python/app.py       # Flask server FAISS
│── Dockerfile          # Docker image build
│── requirements.txt    # Python dependencies
│── config.json         # Config file
└── init.sh             # Init script (start up)