from pymongo import MongoClient
from app.config import settings
from bson.objectid import ObjectId

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
documents_collection = db["documents"]

def insert_document(doc: dict):
    return documents_collection.insert_one(doc)

def get_all_documents():
    return list(documents_collection.find())

def find_document_by_id(doc_id):
    return documents_collection.find_one({"_id": ObjectId(doc_id)})
