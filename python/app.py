from flask import Flask, request, jsonify
from lib.mongo_controller import MongoController
from lib.universal_embedder import UniversalEmbedder
from lib.faiss_controller import FaissController

# Initialize Flask app
app = Flask(__name__)

# Initialize controllers
mongo = MongoController()
embedder = UniversalEmbedder()
faiss = FaissController()

# ----- ENDPOINTS -----
@app.route("/v1/embeddings", methods=["POST"])
def generate():
    """
    Expected request JSON body:
        - input (str): "plain text or base64"
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid request format"}), 400

        # Retrieve data
        data = request_data.get("input")
        
        if not data:
            return jsonify({"error": "'input' param must be provided"}), 400
        # Create embeddings
        print("generating embeddings")
        embedding_result = embedder.embed(request_data, storable=True)
        print(f"generated embeddings {embedding_result}")
        print("building response")
        
        return jsonify({
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": emb
                    }
                    for index, emb in enumerate(embedding_result["bytes"])
                ],
                "object": "list",
                "model": "universal-embedder",
                
            }), 200
    except Exception as e:
        print(f"Error in generate(): {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/v1/embeddings/<string:db>/<string:coll>/save", methods=["POST"])
def save(db, coll):
    """ 
    Expected request JSON body:
        - input (str): "plain text or base64"
        - file_name (str): "file name"
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid request format"}), 400

        data = request_data.get("input")
        name = request_data.get("file_name")

        if not name:
            return jsonify({"error": "Missing required field: file_name"}), 400

        if not data:
            return jsonify({"error": "'input' param must be provided"}), 400

        # Generate embeddings
        embedding_result = embedder.embed(request_data, storable=True)
        bytes_embeddings = embedding_result.get("bytes", [])
        raw_data = embedding_result.get("raw", [])

        if not bytes_embeddings or not raw_data:
            return jsonify({"error": "Failed to generate embeddings"}), 500

        # Insert embeddings into Faiss
        faiss_ids = faiss.add(db, coll, bytes_embeddings)
        if not faiss_ids:
            return jsonify({"error": "Failed to insert embeddings into Faiss"}), 500

        # Prepare document for MongoDB
        document = {
            "name": name,
            "raw": raw_data,
            "faiss_ids": faiss_ids
        }

        # Insert into MongoDB
        mongo.insert(db, coll, document)

        return jsonify({"message": "Data saved successfully", "faiss_ids": faiss_ids}), 200

    except Exception as e:
        print(f"Error in save(): {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/v1/embeddings/<string:db>/<string:coll>/search", methods=["POST"])
def search(db, coll):
    """ 
    Expected request JSON body:
        - input (str): "plain text or base64"
        - k: Number of results to return (default = 5)
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid request format"}), 400

        data = request_data.get("input")
        k = request_data.get("k", 5)

        if not data:
            return jsonify({"error": "'input' param must be provided"}), 400

        # Generate embedding for the search query
        embedding_result = embedder.embed(request_data, storable=True)
        query_embedding = embedding_result.get("bytes", [])

        if not query_embedding:
            return jsonify({"error": "Failed to generate embedding for query"}), 500

        # Search in Faiss
        search_results = faiss.search(db, coll, query_embedding[0], k)  # Query with first embedding
        if not search_results:
            return jsonify({"message": "No matches found"}), 200

        faiss_ids = list(search_results.keys())
        faiss_scores = list(search_results.values())
        results = {}
        # Retrieve documents from MongoDB based on Faiss IDs
        for fid in faiss_ids:
            query = {"faiss_ids": {"$in": [fid]}}
            tmp = mongo.find(db, coll, query)
            if tmp["name"] in results: # If document exists in results, add new element
                results[tmp["name"]].append({
                    "content": tmp["raw"][tmp["faiss_ids"].index(fid)],
                    "score": faiss_scores[faiss_ids.index(fid)]
                })
            else: # create result for document
                results[tmp["name"]] = [
                    {
                        "content": tmp["raw"][tmp["faiss_ids"].index(fid)],
                        "score": faiss_scores[faiss_ids.index(fid)]
                    }
                ]
        return jsonify({"matches": results}), 200

    except Exception as e:
        print(f"Error in search(): {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/v1/embeddings/<string:db>/<string:coll>/delete", methods=["POST"])
def delete(db, coll):
    """ 
    Expected request JSON body:
        - query: JSON string to filter which documents to delete
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
            return jsonify({"error": "Missing required field: query"}), 400

        query = request_data["query"]

        # Find documents in MongoDB that match the query
        documents = mongo.find(db, coll, query)
        if not documents:
            return jsonify({"message": "No matching documents found"}), 200

        # Extract Faiss IDs from matching documents
        faiss_ids = [faiss_id for doc in documents if "faiss_ids" in doc for faiss_id in doc["faiss_ids"]]

        # Delete embeddings from Faiss
        if faiss_ids:
            faiss.delete(db, coll, faiss_ids)

        # Delete documents from MongoDB
        mongo.delete(db, coll, query)

        return jsonify({"message": f"Deleted {len(documents)} documents and {len(faiss_ids)} embeddings"}), 200

    except Exception as e:
        print(f"Error in delete(): {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/v1/embeddings/<string:db>/<string:coll>/update", methods=["POST"])
def update(db, coll):
    """ 
    Expected request JSON body:
        - input (str): "plain text or base64"
        - query (json_str): JSON string to filter which documents to update
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
            return jsonify({"error": "Missing required field: query"}), 400

        query = request_data["query"]
        new_data = request_data.get("input")

        if not new_data:
            return jsonify({"error": "'input' param must be provided"}), 400

        # Find documents in MongoDB that match the query
        documents = mongo.find(db, coll, query)
        if not documents:
            return jsonify({"message": "No matching documents found"}), 200

        # Generate new embeddings
        embedding_result = embedder.embed(request_data, storable=True)
        new_bytes_embeddings = embedding_result.get("bytes", [])
        new_raw_data = embedding_result.get("raw", [])

        if not new_bytes_embeddings or not new_raw_data:
            return jsonify({"error": "Failed to generate embeddings for new data"}), 500

        # Extract old Faiss IDs from documents
        old_faiss_ids = [faiss_id for doc in documents if "faiss_ids" in doc for faiss_id in doc["faiss_ids"]]

        # Replace embeddings in Faiss
        new_faiss_ids = faiss.replace(db, coll, old_faiss_ids, new_bytes_embeddings)
        if not new_faiss_ids:
            return jsonify({"error": "Failed to update embeddings in Faiss"}), 500

        # Update documents in MongoDB
        update_fields = {
            "raw": new_raw_data,
            "faiss_ids": new_faiss_ids
        }
        mongo.update(db, coll, query, update_fields)

        return jsonify({"message": f"Updated {len(documents)} documents", "new_faiss_ids": new_faiss_ids}), 200

    except Exception as e:
        print(f"Error in update(): {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/v1/embeddings/<string:db>/<string:coll>/drop", methods=["POST"])
def drop(db, coll):
    """ 
    Expected request JSON body:
        - pass: "Sanolivar2024*" (required)
        ++
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data or "pass" not in request_data:
            return jsonify({"error": "Missing required field: pass"}), 400

        # Validate password
        if request_data["pass"] != "Sanolivar2024*":
            return jsonify({"error": "Unauthorized"}), 403

        # Delete the collection from MongoDB
        mongo.drop(db, coll)

        # Delete the Faiss index (removes from RAM and disk)
        if faiss.delete_index(db, coll):
            return jsonify({"message": f"Collection {db}/{coll} has been fully deleted"}), 200
        else:
            return jsonify({"message": f"MongoDB collection deleted, but Faiss index not found"}), 200

    except Exception as e:
        print(f"Error in drop(): {e}")
        return jsonify({"error": str(e)}), 500


# ----- START SERVER -----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)