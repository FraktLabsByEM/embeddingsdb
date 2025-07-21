import os
import faiss
import time
import gc  # Garbage collector for memory management
import numpy as np

class FaissController:
    def __init__(self, base_path="/appdata/faiss"):
        """Initialize FaissController with a predefined storage path."""
        self.base_path = base_path  # Base directory for storing indexes
        
        self.indexes = {}  # Dictionary to store references to loaded indexes in RAM

        # Ensure the base directory exists
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path) 
                

    def _import(self, db, collection):
        """Load a Faiss index from disk and store it in RAM."""
        key = f"{db}/{collection}"
        index_path = os.path.join(self.base_path, db, f"{collection}.faiss")

        if not os.path.exists(index_path):
            print(f"Index not found: {index_path}")
            return None  # Return None if the index file does not exist

        print(f"Loading index into RAM: {key}")
        index = faiss.read_index(index_path)  # Load the Faiss index from disk

        # Store the index in memory with a timestamp
        self.indexes[key] = {
            "ref": index,
            "timestamp": time.time()  # Store current timestamp
        }
        return index
    
    
    def _export(self, db, collection):
        """Save a Faiss index from RAM to disk."""
        key = f"{db}/{collection}"

        # Check if the index is loaded in RAM
        if key not in self.indexes:
            print(f"Index not found in RAM: {key}")
            return False

        index = self.indexes[key]["ref"]  # Get the Faiss index reference
        db_path = os.path.join(self.base_path, db)

        # Ensure the database directory exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        index_path = os.path.join(db_path, f"{collection}.faiss")
        faiss.write_index(index, index_path)  # Save the index to disk

        print(f"Index saved: {index_path}")
        return True
    
    
    def unmount(self, db, collection):
        """Remove a Faiss index from RAM to free memory."""
        key = f"{db}/{collection}"

        if key in self.indexes:
            del self.indexes[key]  # Remove the reference
            gc.collect()  # Force garbage collection
            print(f"Index unmounted and memory freed: {key}")
            return True
        else:
            print(f"Index not found in RAM: {key}")
            return False
        
        
    def check_time(self):
        """Return the last interaction timestamp of all mounted indices."""
        if not self.indexes:
            print("No indices are currently mounted.")
            return {}

        timestamps = {
            key: self.indexes[key]["timestamp"] for key in self.indexes
        }

        return timestamps
    
    def create(self, db, collection, neightboors, ln):
        try:
            key = f"{db}/{collection}"
            if key in self.indexes:
                index = self._import(db, collection)
                if index is None:
                    return { "error": f"Index '{db}/{collection}' already exists" }
            print(f"Creating new index for {key}")
            base_index = faiss.IndexHNSWFlat(ln, neightboors)
            base_index.efSearch = 64
            base_index.efConstruction = 80 # Construction "quality"
            index = faiss.IndexIDMap(base_index)
            # Create ids
            self.indexes[key] = {"ref": index, "timestamp": time.time()}
            return True
        except Exception as err:
            print(err)
            return { "error": err }
    
    
    def add(self, db, collection, embeddings):
        try:
            """Add new embeddings to a Faiss index and return assigned IDs."""
            key = f"{db}/{collection}"
            # Ensure the index is loaded in RAM
            if key not in self.indexes:
                return None

            index = self.indexes[key]["ref"]  # Get the Faiss index reference

            # Generate unique IDs for each embedding
            # existing_ids = set(int(index.id_map.at(i)) for i in range(index.id_map.size())) if hasattr(index, "id_map") and index.id_map.size() > 0 else set()
            # new_ids = np.arange(len(existing_ids), len(existing_ids) + len(embeddings))
            new_start = 0
            if hasattr(index, "id_map") and index.id_map.size() > 0:
                existing_ids = set(int(index.id_map.at(i)) for i in range(index.id_map.size()))
                new_start = max(existing_ids) + 1

            new_ids = np.arange(new_start, new_start + len(embeddings), dtype='int64')

            # Convert embeddings to NumPy array and add them to Faiss
            embeddings = np.array(embeddings, dtype=np.float32)
            index.add_with_ids(embeddings, new_ids)

            # Update timestamp
            self.indexes[key]["timestamp"] = time.time()

            print(f"Added {len(embeddings)} embeddings to {key}")
            self._export(db, collection)
            return new_ids.tolist()  # Return assigned IDs
        except Exception as err:
            print(f"Error in faiss.add(): {err}")
            return None


    def search(self, db, collection, query_embedding, k=5):
        """Search for the k most similar embeddings in a Faiss index and return their IDs with distances."""
        key = f"{db}/{collection}"

        # Ensure the index is loaded in RAM
        if key not in self.indexes:
            index = self._import(db, collection)
            if index is None:
                print(f"Index not found: {key}")
                return {}

        index = self.indexes[key]["ref"]  # Get the Faiss index reference

        # Convert query to NumPy array (must be 2D)
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Perform search
        distances, ids = index.search(query, k)

        # Update timestamp
        self.indexes[key]["timestamp"] = time.time()

        # Convert results into a dictionary {id: distance}
        results = {int(id_): float(dist) for id_, dist in zip(ids.flatten(), distances.flatten()) if id_ != -1}

        print(f"Search completed on {key}: {results}")
        return results  # Return dictionary {id: distance}
    
    
    def delete_index(self, db, collection):
        """Delete a Faiss index from disk and unmount it from RAM."""
        key = f"{db}/{collection}"
        index_path = os.path.join(self.base_path, db, f"{collection}.faiss")

        # Ensure the index is unmounted from RAM
        self.unmount(db, collection)

        # Delete index file from disk
        if os.path.exists(index_path):
            os.remove(index_path)
            print(f"Deleted Faiss index: {index_path}")
            return True
        else:
            print(f"Faiss index not found: {index_path}")
            return False