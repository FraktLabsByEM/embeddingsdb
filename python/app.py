import os
import sys

# Append Audioclip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "AudioCLIP")))

import time
from bson import ObjectId
from flask_cors import CORS
from flask import Flask, request, jsonify
from lib.mongo_controller import MongoController
from lib.faiss_controller import FaissController
from lib.universal_embedder import UniversalEmbedder

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize controllers
mongo = MongoController()
faiss = FaissController()
embedder = UniversalEmbedder()

def server_fail(error):
    print(f"Server error -> {error}")
    return jsonify({ "status": "fail", "error": f"Server error -> {error}" }), 501

def user_error(error):
    return jsonify({ "status": "fail", "error": error }), 401

def success(json):
    json["status"] = "ok"
    return jsonify(json), 200

def add(db, coll, input, name):
    try:
        # Generate embeddings
        embedding_result = embedder.embed({ "input": input }, storable=True)
        bytes_embeddings = embedding_result.get("bytes")
        raw_data = embedding_result.get("raw")

        # Insert embeddings into Faiss
        faiss_ids = faiss.add(db, coll, bytes_embeddings)

        # Prepare document for MongoDB
        document = {
            "name": name,
            "source": input,
            "raw": raw_data,
            "faiss_ids": faiss_ids,
            "timestamp": time.time()
        }

        # Insert into MongoDB
        mongo.insert(db, coll, document)
        return faiss_ids
    except Exception as err:
        print(f"flask - add error: {err}")
        return None

# ----- ENDPOINTS -----
@app.route("/v1/embeddings", methods=["POST"])
def generate():
    """
    Generate embeddings from text, image, audio or file
    
    Expected request JSON body:
        - input (str): "plain text or base64"
    """
    try:
        request_data = request.get_json()
        # Validations
        if not request_data or not "input" in request_data:
            return user_error(f"Missing required field: 'input'")
        # Generate embeddings
        embedding_result = embedder.embed(request_data, storable=True)
        # Build result
        result = {
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": emb
                    }
                    for index, emb in enumerate(embedding_result["bytes"])
                ],
                "object": "list",
                "model": "snlr-universal-embedder"
            }
        return success(result)
    except Exception as err:
        print(f"Error in generate(): {err}")
        return server_fail(err)
    

@app.route("/v1/embeddings/<string:db>/<string:coll>/create", methods=["POST"])
def create_index(db, coll):
    """
    Create an empty faiss-mongo peer index
    
    Expected JSON body:
        - cluster_size (int): Cluster size
    """
    try:
        request_data = request.get_json()
        # Validations
        if not request_data:
            user_error("Invalid request format")
        if not 'cluster_size' in request_data:
            user_error("Missing required field: 'cluster_size'")
        # Process
        cluster_size = request_data.get("cluster_size")
        result = faiss.create(db, coll, cluster_size)
        if result is True:
            return success({ "message": f"Index {db}/{coll} created successfully"})
        server_fail(f"Unknown error while creating index.")
    except Exception as err:
        print(f"Error in create_index(): {err}")
        return server_fail(err)
    

@app.route("/v1/embeddings/<string:db>/<string:coll>/view", methods=["POST"])
def view(db, coll):
    """
    View collection entries
    """
    try:
        response = mongo.find_all(db, coll)
        return success(response)
    except Exception as err:
        print(f"Error in view(): {err}")
        return server_fail(err)
    

@app.route("/v1/embeddings/<string:db>/<string:coll>/save", methods=["POST"])
def save(db, coll):
    """ 
    Save entry from text, image, audio or file
    
    Expected request JSON body:
        - file_name (str): File name
        - input (str): plain text or base64
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data:
            return user_error("Invalid request format")
        if not "input" in request_data:
            return user_error("Missing required field: 'input'.")
        if not "file_name" in request_data:
            return user_error("Missing required field: 'file_name'.")
            
        data = request_data.get("input")
        name = request_data.get("file_name")
        faiss_ids = add(db, coll, data, name)

        if faiss_ids is None:
            return server_fail("Unknown error while trying to save embeddings")
        
        return success({ "message": "Data saved successfully", "ids": faiss_ids, "file_name": name })

    except Exception as err:
        print(f"Error in save(): {err}")
        return server_fail(str(err))


@app.route("/v1/embeddings/<string:db>/<string:coll>/search", methods=["POST"])
def search(db, coll):
    """ 
    Reverse semantic search
    
    Expected request JSON body:
        - input (str): plain text or base64
        - results: Number of results to return (default = 5)
    """
    try:
        request_data = request.get_json()
        # Validate request JSON
        if not request_data:
            return user_error("Invalid request format")
        if not "input" in request_data:
            return user_error("Missing required field: 'input'")
        
        k = request_data.get("results", 5)

        # Generate embedding for the search query
        embedding_result = embedder.embed(request_data, storable=True)
        query_embedding = embedding_result.get("bytes", [])

        if not query_embedding:
            return server_fail("Failed to generate embeddings.")

        # Search in Faiss
        search_results = {}
        for emb in query_embedding:
            res = faiss.search(db, coll, emb, k)
            if res.keys() > 0:
                for key in res.keys():
                    search_results[key] = res[key]
                    
        # Filter k results
        top_k = dict(sorted(search_results.items(), key=lambda item: item[1])[:k])
        faiss_ids = list(top_k.keys())
        faiss_scores = list(top_k.values())

        # Retrieve documents from MongoDB based on Faiss IDs
        results = {}
        for fid in faiss_ids:
            query = {"faiss_ids": {"$in": [fid]}}
            tmp = mongo.find(db, coll, query)
            if tmp is not None:
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
        
        return success({"matches": results})

    except Exception as err:
        print(f"Error in search(): {err}")
        return server_fail(err)



@app.route("/v1/embeddings/<string:db>/<string:coll>/delete", methods=["POST"])
def delete(db, coll):
    """ 
    Delete entry
    
    Expected request JSON body:
        - id: Entry id
    """
    try:
        request_data = request.get_json()
        # Validate request JSON
        if not request_data or "id" not in request_data:
            return user_error("Missing required field: 'id'")

        id = request_data["id"]
        
        # Confirm the entry exists
        _doc = mongo.find(db, coll, { "_id": ObjectId(id) })
        if _doc is None:
            return user_error({ "message": f"Entry with id '{id}' not found in '{db}/{coll}'." })
        
        # Fetch all documents
        full_docs = mongo.find_all(db, coll)
        
        # Drop faiss index
        cluster_size = faiss.get_clustersize(db, coll)
        faiss.delete_index(db, coll)
        # Create a new one
        faiss.create(db, coll, cluster_size)
        
        total = 0
        succesfully = 0
        deleted_embeddings = 0
        # Remove deleted document
        for doc in full_docs:
            # Document query
            q = { "_id": ObjectId(doc["_id"]) }
            # Ignored entries
            if doc['id'] != id:
                total += 1 # Increase total count
                # Generate new embeddings
                tmp_res = embedder.embed({ "input": doc["source"]}, True)
                # Get embeddings result
                tmp_raw = tmp_res.get("raw")
                tmp_emb = tmp_res.get("bytes")
                # Insert into faiss index
                tmp_ids = faiss.add(db, coll, tmp_emb)
                # update on mongo
                updated = mongo.update(db, coll, q, {
                        "raw": tmp_raw,
                        "faiss_ids": tmp_ids
                    })
                # Confirm update
                if updated: succesfully += 1
            # Deleted entries
            else:
                deleted_embeddings += len(doc["faiss_ids"])
                # Remove from mongo
                mongo.delete(db, coll, q)
        
        if total == succesfully:
            return success({ "message": f"Deleted 1 entry with {deleted_embeddings} embeddings."})
        else:
            return success({ 
                    "message": f"Entry deleted succesfully! But something went wrong indexing other entries. {succesfully}/{total} reindexed.",
                    "location": f"{db}/{coll}",
                    "previous_data": full_docs
                })

    except Exception as err:
        print(f"Error in delete(): {err}")
        return server_fail(err)

@app.route("/v1/embeddings/<string:db>/<string:coll>/update", methods=["POST"])
def update(db, coll):
    """ 
    Update entry data
    
    Expected request JSON body:
        - id (str): Entry id
        - file_name (str): File name
        - input (str): plain text or base64
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        
        # Validate params
        required_fields = {"id", "file_name", "input"}
        if not request_data or not required_fields.issubset(request_data):
            return user_error(f"Missing some of the required fields: id, file_name, input")
        
        id = request_data.get("id")
        file_name = request_data.get("file_name")
        new_data = request_data.get("input")
        
        # Confirm the entry exists
        _doc = mongo.find(db, coll, { "_id": ObjectId(id) })
        if _doc is None:
            return user_error({ "message": f"Entry with id '{id}' not found in '{db}/{coll}'." })
        
        # Fetch all documents
        full_docs = mongo.find_all(db, coll)
        
        # Drop faiss index
        cluster_size = faiss.get_clustersize(db, coll)
        faiss.delete_index(db, coll)
        # Create a new one
        faiss.create(db, coll, cluster_size)
        
        total = 0
        succesfully = 0
        updated_embeddings = 0
        # Remove deleted document
        for doc in full_docs:
            total += 1 # Increase total count
            # Document query
            q = { "_id": ObjectId(doc["_id"]) }
            # Generate new embeddings
            tmp_res = embedder.embed({ "input": doc["source"] if doc['id'] != id else new_data }, True)
            # Get embeddings result
            tmp_raw = tmp_res.get("raw")
            tmp_emb = tmp_res.get("bytes")
            # Insert into faiss index
            tmp_ids = faiss.add(db, coll, tmp_emb)
            # update on mongo
            updated = mongo.update(db, coll, q, {
                    "file_name": doc["file_name"] if doc["id"] != id else file_name,
                    "raw": tmp_raw,
                    "faiss_ids": tmp_ids
                })
            # Confirm update
            if updated: succesfully += 1
            # Save new updated embeddings len
            if doc["id"] == id:
                updated_embeddings = len(tmp_emb)
        
        if total == succesfully:
            return success({ "message": f"Updated 1 entry with {updated_embeddings} embeddings."})
        else:
            return success({ 
                    "message": f"Entry updated succesfully! But something went wrong indexing other entries. {succesfully}/{total} reindexed.",
                    "location": f"{db}/{coll}",
                    "previous_data": full_docs
                })
        
    except Exception as err:
        print(f"Error in update(): {err}")
        return server_fail(err)



@app.route("/v1/embeddings/<string:db>/<string:coll>/drop", methods=["POST"])
def drop(db, coll):
    """ 
    Drop entry collection
    
    Expected request JSON body:
        - pass: "Sanolivar2024*" (required)
    """
    try:
        # Validate request JSON
        request_data = request.get_json()
        if not request_data or "pass" not in request_data:
            return jsonify({"error": "Missing required field: pass"}), 400

        # Validate password
        if request_data["pass"] != "Sanolivar2024*":
            return user_error("Unauthorized")

        # Delete the collection from MongoDB
        mongo.drop_cl(db, coll)

        # Delete the Faiss index (removes from RAM and disk)
        if faiss.delete_index(db, coll):
            return success({"message": f"Collection '{db}/{coll}' has been deleted"})
        else:
            return success({"message": f"Index '{db}/{coll}' not found."})

    except Exception as err:
        print(f"Error in drop(): {err}")
        return server_fail(err)

# ----- START SERVER -----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)