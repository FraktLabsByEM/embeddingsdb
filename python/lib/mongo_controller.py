from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

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
        except PyMongoError as err:
            raise ValueError(f"drop_db - Error while droping db '{db}': {err}")

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
        except PyMongoError as err:
            raise ValueError(f"create_cl - Error while creating collection '{collection_name}' into '{db}': {err}")

    def drop_cl(self, db, collection_name):
        """
        Remove a collection from database
        :param db: Database name
        :param collection_name: Collection name
        """
        try:
            self.client[db].drop_collection(collection_name)
            return True
        except PyMongoError as err:
            raise ValueError(f"drop_cl - Error while droping collection '{collection_name}' from '{db}': {err}")

    def find_all(self, db, collection_name):
        """
        Retrieve all documents from collection.
        :param db: Database name
        :param collection_name: Collection name
        :return: Results list
        """
        try:
            collection = self.client[db][collection_name]
            documents = list(collection.find())
            for doc in documents:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            return documents
        except PyMongoError as err:
            raise ValueError(f"find_all - Error while trying to find documents on '{db}/{collection_name}': {err}")

    def find(self, db, collection_name, query):
        """
        Find a document.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :return: First document found matching the query
        """
        try:
            collection = self.client[db][collection_name]
            document = collection.find_one(query)
            if not document:
                print(f"find - No results on '{db}'/'{collection_name}'")
                return None
            document["id"] = str(document["_id"])
            del document["_id"]
            return document
        except PyMongoError as err:
            raise ValueError(f"find - Error while trying to find a document on '{db}/{collection_name}': {err}")

    def insert(self, db, collection_name, document):
        """
        Insert a document into collection
        :param db: Database name
        :param collection_name: Collection name
        :param document: Documento to insert
        :return: Document id
        """
        try:
            collection = self.client[db][collection_name]
            result = collection.insert_one(document)
            return result.inserted_id
        except DuplicateKeyError as derr:
            print(derr)
            raise ValueError(f"insert - Entry already exists")
        except PyMongoError as err:
            raise ValueError(f"insert - Error while trying to insert a document on '{db}/{collection_name}': {err}")
            
    def update(self, db, collection_name, query, new_values):
        """
        Update document from collection.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :param new_values: New values
        :return: Number of documents changed
        """
        try:
            collection = self.client[db][collection_name]
            result = collection.update_one(query, {'$set': new_values})
            if result.matched_count > 0:
                if result.modified_count > 0:
                    return True
                raise ValueError(f"update - No updates applied on '{db}/{collection_name}'. Data up to date.")
            raise ValueError(f"update - No match found on '{db}/{collection_name}'.")
        except PyMongoError as err:
            raise ValueError(f"update - Error while trying to update a document on '{db}/{collection_name}': {err}")

    def delete(self, db, collection_name, query):
        """
        Delete a document from collection.
        :param db: Database name
        :param collection_name: Collection name
        :param query: Search filter
        :return: Number of deleted documents
        """
        try:
            collection = self.client[db][collection_name]
            result = collection.delete_one(query)
            if result.deleted_count == 1:
                return True
            raise ValueError(f"delete - There's no entry matching your query on '{db}/{collection_name}'")
        except PyMongoError as err:
            raise ValueError(f"delete - Error while trying to delete a document on '{db}/{collection_name}': {err}")