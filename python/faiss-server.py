import os
import time
from flask_cors import CORS
from flask import Flask, request, jsonify
from lib.faiss_controller import FaissController
from lib.mongo_controller import MongoController
from lib.universal_embedder import UniversalEmbedder

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize controllers
mongo = MongoController()
embedder = UniversalEmbedder()
faiss = FaissController()

# Retrieve Admin pass
ADMIN_PASS = os.getenv("ADMIN_PASS") or "snlrdev2025"


# ----------------------------------------
# --------------- Response standards -----
# ----------------------------------------

def user_error(error):
    """
    Args:
        error (str): Error message
    """
    print(f"[USER_ERROR] -> {error}")
    return jsonify({ "status": "fail", "error": error }), 400

def server_error(error):
    """
    Args:
        error (str): Error message
    """
    print(f"[SERVER_ERROR] -> {error}")
    return jsonify({ "status": "fail", "error": f"Server error: {error}" }), 500

def success(response):
    """
    Args:
        response (dict): Response object
    """
    response["status"] = "ok"
    return jsonify(response), 200



def add(db, coll, input, name):
    try:
        # Generate embeddings
        embedding_result = embedder.embed({ "input": input }, storable=True)
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
        return faiss_ids, False
    except Exception as err:
        return False, err

# ----------------------------------------
# --------------- ENDPOINTS --------------
# ----------------------------------------
@app.route("/v1/embeddings", methods=["POST"])
def generate():
    """
    Generate embeddings
    ---
    Args:
        input (str): plain text or base64
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data:
            return user_error("Invalid request format.")
        if not "data" in request_data:
            return user_error("Missing required parameter 'input'.")
        # Create embeddings
        print("generating embeddings")
        embedding_result = embedder.embed(request_data, storable=True)
        
        return success({
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
        })
        
    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------

@app.route("/v1/embeddings/<string:db>/<string:coll>/create", methods=["POST"])
def create_index(db, coll):
    """
    Create a new empty faiss index.
    ---
    Args:
        neighboors (int): Number of neighboors per cluster
        length (int): Embedings dimension or length
    """
    try:
        request_data = request.get_json()
        # Validate params
        if not request_data:
            return user_error("Invalid request format.")
        if not "neighboors" in request_data:
            return user_error("Missing required parameter 'neighboors'.")
        if not "length" in request_data:
            return user_error("Missing required parameter 'length'.")
        # Retrievve params
        neighboors = request_data.get("neighboors")
        ln = request_data.get("length")
        # Create index
        result = faiss.create(db, coll, neighboors, ln)

        if result:
            return success({"message": f"Index '{db}/{coll}' created successfully"})
        else:
            return server_error("Unknown error occurred during index creation")

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------
    

@app.route("/v1/embeddings/<string:db>/<string:coll>/save", methods=["POST"])
def save(db, coll):
    """ 
    Save new embeddings
    ---
    Args:
        input (str): Plain text or base64
        file_name (str): File name or identifier
    """
    try:
        request_data = request.get_json()
        # Validate params
        if not request_data:
            return user_error("Invalid request format")
        if not "input" in request_data:
            return user_error("Missing required parameter 'input'.")
        if not "file_name" in request_data:
            return user_error("Missing required parameter 'file_name'.")

        data = request_data.get("input")
        name = request_data.get("file_name")

        faiss_ids = add(db, coll, data, name)

        if faiss_ids:
            return success({"message": "Data saved successfully", "faiss_ids": faiss_ids})
        else:
            return server_error(f"Unknown error occurred while trying to add a new element on '{db}/{coll}'.")

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------


@app.route("/v1/embeddings/<string:db>/<string:coll>/search", methods=["POST"])
def search(db, coll):
    """
    Semantic search on an index
    ---
    Args:
        input (str): plain text or base64
        k: Number of results to return (default = 5)
    """
    try:
        request_data = request.get_json()
        # Validate params
        if not request_data:
            return user_error("Invalid request format")
        if not "input" in request_data:
            return user_error("Missing required parameter 'input'.")
        # Retrieve param
        k = request_data.get("k", 5)
        start_time = time.time()

        # Generate embedding for the search query
        embedding_result = embedder.embed(request_data, storable=True)
        query_embedding = embedding_result.get("bytes", [])

        if not query_embedding:
            return server_error(f"Failed to generate embedding for reverse search on '{db}/{coll}'")

        # Search in Faiss
        search_results = faiss.search(db, coll, query_embedding[0], k)  # Query with first embedding
        
        # Validate results
        if not search_results:
            end_time = time.time()
            return success({ "matches": {}, "message": "No matches found.", "latency": end_time - start_time })

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
        
        # Build response
        end_time = time.time()
        return success({"matches": results, "message": f"{len(faiss_ids)} matches found.", "latency": end_time - start_time})

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------


@app.route("/v1/embeddings/<string:db>/<string:coll>/delete", methods=["POST"])
def delete(db, coll):
    """
    Delete embedding from db
    ---
    Args:
        file_name (str): File name or identifier
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data or "file_name" not in request_data:
            return user_error("Missing required parameter 'file_name'.")

        filename = request_data["file_name"]
        query = { "name": filename }
        
        # Fetch all documents from mongo
        full_docs = mongo.find_all(db, coll)
        
        deleted_embeddings = 0
        # Drop faiss index
        emb_deleted =  faiss.delete_index(db, coll)
        
        updated_ok = True
        # Remove deleted document
        for doc in full_docs:
            if 'name' in doc and doc['name'] != filename:
                # Loop thru all raw chunks
                new_emb = []
                for ind, rw in enumerate(doc["raw"]):
                    # Generate embedings
                    emb = embedder.embed({"input": rw }, storable=True)
                    # Add embeddings
                    new_emb.append(emb["bytes"][0])
                # Update faiss id
                new_ids = faiss.add(db, coll, new_emb)
                # Update in mongo
                updated_ok = mongo.update(db, coll, { "name": doc["name"] }, { "faiss_ids": new_ids })
                if not updated_ok:
                    break
            elif 'name' in doc:
                deleted_embeddings = len(doc["faiss_ids"])
        
        # Remove document
        deleted = mongo.delete(db, coll, query)
        # Update mongo documents
        if emb_deleted and updated_ok and deleted:
            return success({"message": f"Deleted 1 document and {deleted_embeddings} embeddings"})
        else:
            return server_error(f"Unknown error occurred while trying to remove an element on '{db}/{coll}'.")

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------



@app.route("/v1/embeddings/<string:db>/<string:coll>/update", methods=["POST"])
def update(db, coll):
    """
    Update embeddings on index.
    ---
    Args:
        input (str): plain text or base64
        file_name (str): File name or identifier
    """
    try:
        request_data = request.get_json()
        # Validate request params
        if not request_data:
            return user_error("Invalid request format.")
        if not "input" in request_data:
            return user_error("Missing required parameter 'input'.")
        if not "file_name" in request_data:
            return user_error("Missing required parameter 'file_name'.")

        file_name = request_data["file_name"]
        new_data = request_data["input"]
        
        # Mongo query
        query = { "name": file_name }

        # Fetch all documents
        full_docs = mongo.find_all(db, coll)
        
        # Drop faiss index
        emb_deleted =  faiss.delete_index(db, coll)
        
        updated_ok = True
        # Remove deleted document while recreating index
        for doc in full_docs:
            if 'name' in doc and doc['name'] != file_name:
                # Loop thru all raw chunks
                new_emb = []
                for ind, rw in enumerate(doc["raw"]):
                    # Generate embedings
                    emb = embedder.embed({"input": rw }, storable=True)
                    # Add embeddings
                    new_emb.append(emb["bytes"][0])
                # Update faiss id
                new_ids = faiss.add(db, coll, new_emb)
                # Update in mongo
                updated_ok = mongo.update(db, coll, { "name": doc["name"] }, { "faiss_ids": new_ids })
                if not updated_ok:
                    break
        
        # Remove document
        deleted = mongo.delete(db, coll, query)
        # Add new document
        fids = add(db, coll, {"input": new_data}, file_name)
        
        if emb_deleted and updated_ok and deleted:
            return success({"message": f"Updated 1 document with {len(fids)} embeddings"})
        else:
            return server_error(f"Unknown error occurred while trying to remove an element on '{db}/{coll}'.")

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------



@app.route("/v1/embeddings/<string:db>/<string:coll>/drop", methods=["POST"])
def drop(db, coll):
    """ 
    Drop faiss index
    ---
    Args:
        pass: Administrator password
    """
    try:
        request_data = request.get_json()
        # Validate request params
        if not request_data:
            return user_error("Invalid request format.")
        if not "pass" in request_data:
            return user_error("Missing required parameter 'pass'.")

        # Validate password
        if request_data["pass"] != ADMIN_PASS:
            return user_error("Unauthorized")

        # Delete the collection from MongoDB
        mongo.drop_cl(db, coll)

        # Delete the Faiss index (removes from RAM and disk)
        if faiss.delete_index(db, coll):
            return success({"message": f"Index '{db}/{coll}' has been fully deleted."})
        else:
            return server_error(f"Unknown error occurred while trying to remove index '{db}/{coll}'.")

    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------

@app.route("/v1/embeddings/<string:db>/<string:coll>/raw", methods=["POST"])
def raw(db, coll):
    """
    Get raw data from index
    ---
    """
    try:
        start_time = time.time()
        result = mongo.find_all(db, coll)
        end_time = time.time()
        return success({"matches": result, "message": f"{len(result)} matches found.", "latency": end_time - start_time})
    except Exception as err:
        return server_error(err)
# -------------------------------------------------------------------


# ----- START SERVER -----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3247, debug=True)