from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from flask import jsonify

class MongoController:
    def __init__(self, host='localhost', port=27017):
        """
        Constructor for db connection
        ---
        Args:
            host (str): Address to MongoDB server (default localhost)
            port (str): Port to MongoDB server (default 27017)
        """
        self.client = MongoClient(host, port)

    def drop_db(self, db):
        """
        Drop database.
        ---
        Args:
            db (str): Database name
        """
        try:
            self.client.drop_database(self.client[db])
            return True
        except PyMongoError as e:
            return False

    def create_cl(self, db, collection_name):
        """
        Create a new collection in a db.
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
        """
        
        try:
            self.client[db].create_collection(collection_name)
            return True
        except PyMongoError as err:
            raise ValueError(f"MongoController error on create_cl: {err}")

    def drop_cl(self, db, collection_name):
        """
        Remove a collection from database
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
        """
        try:
            self.client[db].drop_collection(collection_name)
            return True
        except PyMongoError as err:
            raise ValueError(f"MongoController error on create_cl: {err}")

    def find_all(self, db, collection_name):
        """
        Retrieve all documents from collection.
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
        :return: Results list
        """
        try:
            collection = self.client[db][collection_name]
            documents = list(collection.find())
            if documents:
                for doc in documents:
                    doc["_id"] = str(doc["_id"])
                return documents
            else:
                raise ValueError(f"MongoController error on find_all: No documents found on collection '{db}/{collection_name}'")
        except PyMongoError as err:
            raise ValueError(f"MongoController error on find_all: {err}")

    def find(self, db, collection_name, query):
        """
        Find a document.
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
            query (str): Search filter
        :return: First document found matching the query
        """
        collection = self.client[db][collection_name]
        try:
            document = collection.find_one(query)
            if document:
                return document
            else:
                raise ValueError(f"MongoController error on find: No documents found on collection '{db}/{collection_name}' matching your query.")
        except PyMongoError as err:
            raise ValueError(f"MongoController error on find: {err}")

    def insert(self, db, collection_name, document):
        """
        Insert a document into collection
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
            document (str): Documento to insert
        :return: Document id
        """
        collection = self.client[db][collection_name]
        try:
            result = collection.insert_one(document)
            return result.inserted_id
        except DuplicateKeyError as e:
            raise ValueError(f"MongoController error on insert: Duplicated key error.")
        except PyMongoError as err:
            raise ValueError(f"MongoController error on insert: {err}")
            
    def update(self, db, collection_name, query, new_values):
        """
        Update document from collection.
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
            query (str): Search filter
            new_values (str): New values
        :return: Number of documents changed
        """
        collection = self.client[db][collection_name]
        try:
            collection.update_one(query, {'$set': new_values})
            return True
        except PyMongoError as err:
            raise ValueError(f"MongoController error on update: {err}")

    def delete(self, db, collection_name, query):
        """
        Delete a document from collection.
        ---
        Args:
            db (str): Database name
            collection_name (str): Collection name
            query (str): Search filter
        :return: Number of deleted documents
        """
        collection = self.client[db][collection_name]
        try:
            result = collection.delete_one(query)
            return True
        except PyMongoError as err:
            raise ValueError(f"MongoController error on delete: {err}")