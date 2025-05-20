from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from flask import jsonify

class MongoController:
    def __init__(self, host='localhost', port=27017):
        """
        Constructor for db connection
        :param host: Address to MongoDB server (default localhost)
        :param port: Port to MongoDB server (default 27017)
        """
        self.client = MongoClient(host, port)

    def drop_db(self, db):
        """
        Drop database.
        :param db: Database name
        """
        try:
            self.client.drop_database(self.client[db])
            return True
            # return jsonify({ "status": "ok", "code": 200, "message": f"Database '{db}' succesfully droped."})
        except PyMongoError as e:
            return False
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to retrieve documents from collection '{db}.{collection_name}' error: {e}" })

    def create_cl(self, db, collection_name):
        """
        Create a new collection in a db.
        :param db: Database name
        :param collection_name: Collection name
        :return: JSON response
        """
        
        try:
            self.client[db].create_collection(collection_name)
            return True
            # return jsonify({ "status": "ok", "code": 200, "message": f"Collection '{db}.{collection_name}' succesfully created." })
        except PyMongoError as e:
            return False
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to create collection '{db}.{collection_name}' error: {e}" })

    def drop_cl(self, db, collection_name):
        """
        Remove a collection from database
        :param db: Database name
        :param collection_name: Collection name
        """
        try:
            self.client[db].drop_collection(collection_name)
            return True
            # return jsonify({ "status": "ok", "code": 200, "message": f"Collection '{db}.{collection_name}' succesfully dropped." })
        except PyMongoError as e:
            return False
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to drop collection '{db}.{collection_name}' error: {e}" })

    def find_all(self, db, collection_name):
        """
        Retrieve all documents from collection.
        :param db: Database name
        :param collection_name: Collection name
        :return: Results list
        """
        collection = self.client[db][collection_name]
        try:
            documents = list(collection.find())
            if documents:
                for doc in documents:
                    doc["_id"] = str(doc["_id"])
                return documents
            else:
                return None
                # return jsonify({ "status": "fail", "code": 200, "message": f"Nothing found in '{db}.{collection_name}'." })
        except PyMongoError as e:
            return None
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to retrieve documents from collection '{db}.{collection_name}' error: {e}" })

    def find(self, db, collection_name, query):
        """
        Find a document.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :return: First document found matching the query
        """
        collection = self.client[db][collection_name]
        try:
            document = collection.find_one(query)
            if document:
                return document
            else:
                return None
                # return jsonify({ "status": "ok", "code": 300, "message": f"Document not found in '{db}.{collection_name}'"})
        except PyMongoError as e:
            return None
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to find a document from collection '{db}.{collection_name}' error: {e}" })

    def insert(self, db, collection_name, document):
        """
        Insert a document into collection
        :param db: Database name
        :param collection_name: Collection name
        :param document: Documento to insert
        :return: Document id
        """
        collection = self.client[db][collection_name]
        try:
            result = collection.insert_one(document)
            return { "status": True, "id": str(result.inserted_id) }
            # return jsonify({ "status": "ok", "code": 200, "message": f"Document succesfully inserted into '{db}.{collection_name}'.", "doc_id": str(result.inserted_id) })
        except DuplicateKeyError as e:
            return { "status": False, "error": e }
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to isert into collection '{db}.{collection_name}' error: {e}" })
        except PyMongoError as e:
            return { "status": False, "error": e }
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to insert into collection '{db}.{collection_name}' error: {e}" })
            
    def update(self, db, collection_name, query, new_values):
        """
        Update document from collection.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :param new_values: New values
        :return: Number of documents changed
        """
        collection = self.client[db][collection_name]
        try:
            result = collection.update_one(query, {'$set': new_values})
            return True
            # return jsonify({ "status": "ok", "code": 200, "message": f"{result.modified_count } Document(s) succesfully updated in '{db}.{collection_name}'.", "count": result.modified_count })
        except PyMongoError as e:
            return False
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to update document from collection '{db}.{collection_name}' error: {e}" })

    def delete(self, db, collection_name, query):
        """
        Delete a document from collection.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :return: Number of deleted documents
        """
        collection = self.client[db][collection_name]
        try:
            result = collection.delete_one(query)
            return True
            # return jsonify({ "status": "ok", "code": 200, "message": f"{result.deleted_count } Document(s) succesfully deleted in '{db}.{collection_name}'.", "count": result.deleted_count })
        except PyMongoError as e:
            return False
            # return jsonify({ "status": "fail", "code": 300, "message": f"Failed to update document from collection '{db}.{collection_name}' error: {e}" })