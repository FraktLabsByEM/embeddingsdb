import os
import gc  # Garbage collector for memory management
import time
import json
import faiss
import threading
import numpy as np

class FaissController:
    def __init__(self, base_path="/appdata/faiss", size=512):
        """Initialize FaissController with a predefined storage path."""
        self.base_path = base_path  # Base directory for storing indexes
        
        self.indexes = {}  # Dictionary to store references to loaded indexes in RAM
        
        self.size = size # Embeddings dimensionality

        # Ensure the base directory exists
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path) 
                

    def _import(self, db, collection):
        """Load a Faiss index from disk and store it in RAM."""
        try:
            key = f"{db}/{collection}"
            index_path = os.path.join(self.base_path, db, f"{collection}.faiss")

            if not os.path.exists(index_path):
                return None

            index = faiss.read_index(index_path)  # Load the Faiss index from disk

            # Store the index in memory with a timestamp
            self.indexes[key] = {
                "ref": index,
                "timestamp": time.time()  # Store current timestamp
            }
            print(f"Loaded index : {key}")
            return index
        except Exception as err:
            print(f"Faiss import error {err}")
            return None
    
    
    def _export(self, db, collection):
        """Save a Faiss index from RAM to disk."""
        index = None; index_path = None
        try:
            key = f"{db}/{collection}"

            # Check if the index is loaded in RAM
            if key not in self.indexes:
                raise ValueError(f"Faiss index '{key}' not loaded.")

            index = self.indexes[key]["ref"]  # Get the Faiss index reference
            db_path = os.path.join(self.base_path, db)

            # Ensure the database directory exists
            if not os.path.exists(db_path):
                os.makedirs(db_path)

            index_path = os.path.join(db_path, f"{collection}.faiss")

            print(f"Index saved: {key}")
        except Exception as err:
            raise ValueError(f"Error while trying to export faiss index '{key}': {err}")
        finally:
            if index and index_path:
                try:
                    threading.Thread(target=faiss.write_index, args=(index, index_path), daemon=True).start()
                except Exception as ferr:
                    print(f"Error on export thread: {ferr}")
    
    
    def unmount(self, db, collection):
        """Remove a Faiss index from RAM to free memory."""
        try:
            key = f"{db}/{collection}"

            if key in self.indexes:
                del self.indexes[key]  # Remove the reference
                print(f"Index unmounted and memory freed: {key}")
            else:
                print(f"Index not found in RAM: {key}")
        except Exception as err:
            raise ValueError(f"Error while trying to unmount faiss index '{key}': {err}")
        finally:
            # Force garbage collection
            threading.Thread(target=gc.collect, daemon=True).start()
        
        
    def check_time(self):
        """Return the last interaction timestamp of all mounted indices."""
        try:
            if not self.indexes:
                raise ValueError("No indices are currently mounted.")

            timestamps = {
                key: self.indexes[key]["timestamp"] for key in self.indexes
            }

            return timestamps
        except Exception as err:
            raise ValueError(f"Error while trying to check faiss indexes lifetime: {err}")
    
    def create(self, db, collection, cluster_size):
        try:
            key = f"{db}/{collection}"
            if key in self.indexes:
                raise ValueError(f"Index '{key}' already exists")
            else:
                index = self._import(db, collection)
                if index is not None:
                    raise ValueError(f"Index '{key}' already exists")
                
            print(f"Creating new index for {key}")
            base_index = faiss.IndexHNSWFlat(self.size, cluster_size)
            base_index.efSearch = 64
            base_index.efConstruction = 80 # Construction "quality"
            index = faiss.IndexIDMap(base_index)
            # Create ids
            self.indexes[key] = { "ref": index, "timestamp": time.time() }
            return True
        except Exception as err:
            raise ValueError(f"Error while trying to create faiss index '{key}': {err}")
        finally:
            threading.Thread(target=self._save_clustersize, args=(db, collection, cluster_size), daemon=True).start()
            self._export(db, collection)
    
    
    def add(self, db, collection, embeddings):
        try:
            """Add new embeddings to a Faiss index and return assigned IDs."""
            
            # Validation
            if embeddings.shape[0] != 512:
                raise ValueError("Invalid embedding size, expected 512 dimensions.")
            key = f"{db}/{collection}"
            # Ensure the index is loaded in RAM
            if key in self.indexes:
                index = self.indexes[key]["ref"]  # Get the Faiss index reference
            else:
                index = self._import(db, collection)
                if index is None:
                    raise ValueError(f"Failed to insert index into '{key}'. Index not found error.")

            # Generate unique IDs for each embedding
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
            return new_ids.tolist()  # Return assigned IDs
        except Exception as err:
            raise ValueError(f"Error while trying to add vector into faiss index '{key}': {err}")
        finally:
            # Export index
            threading.Thread(target=self._export, args=(db, collection), daemon=True).start()


    def search(self, db, collection, query_embedding, k=5):
        """Search for the k most similar embeddings in a Faiss index and return their IDs with distances."""
        try:
            key = f"{db}/{collection}"

            # Ensure the index is loaded in RAM
            if key in self.indexes:
                index = self.indexes[key]["ref"]  # Get the Faiss index reference
            else:
                index = self._import(db, collection)
                if index is None:
                    raise ValueError(f"Index not found: {key}")

            # Convert query to NumPy array (must be 2D)
            query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            # Perform search
            distances, ids = index.search(query, k)

            # Convert results into a dictionary {id: distance}
            results = {int(id_): float(dist) for id_, dist in zip(ids.flatten(), distances.flatten()) if id_ != -1}

            print(f"Search completed on {key}: {len(results.keys())}")
            return results  # Return dictionary {id: distance}
        except Exception as err:
            raise ValueError(f"Error while trying to search in faiss index '{key}': {err}")
        finally:
            # Update timestamp
            self.indexes[key]["timestamp"] = time.time()
            
    
    
    def delete_index(self, db, collection):
        """Delete a Faiss index from disk and unmount it from RAM."""
        try:
            key = f"{db}/{collection}"
            index_path = os.path.join(self.base_path, db, f"{collection}.faiss")
            
            exists = False
            if os.path.exists(index_path):
                exists = True
                print(f"Deleted Faiss index: {index_path}")
                return True
            else:
                raise ValueError(f"Error while trying to delete faiss index '{key}': The index doesn't exists.")
        except Exception as err:
            raise ValueError(f"Error while trying to delete faiss index '{key}': {err}")
        finally:
            if exists:
                # Ensure the index is unmounted from RAM
                self.unmount(db, collection)

                # Delete index file from disk
                os.remove(index_path)

    def _save_clustersize(self, db, coll, cluster_size):
        pth = os.path.join(self.base_path, "cluster-sizes.json")
        key = f"{db}/{coll}"
        try:
            # If the file exists load it
            if os.path.exists(pth):
                with open(pth, "r") as f:
                    structure = json.load(f)
            else:
                structure = {}
            # Update cluster_size
            structure[key] = cluster_size
            # Update file
            with open(pth, "w") as f:
                json.dump(structure, f, indent=4)
        except Exception as err:
            raise ValueError(f"Error saving structure metadata for '{key}': {err}")
        
    def get_clustersize(self, db, coll):
        pth = os.path.join(self.base_path, "cluster-sizes.json")
        key = f"{db}/{coll}"
        try:
            # Validate file exists
            if not os.path.exists(pth):
                raise ValueError(f"No structure file found at '{pth}'")
            # Open file
            with open(pth, "r") as f:
                structure = json.load(f)
            # Validate existing key
            if key not in structure:
                raise ValueError(f"No cluster size metadata found for '{key}'")
            return structure[key]

        except Exception as err:
            raise ValueError(f"Error loading cluster size for '{key}': {err}")