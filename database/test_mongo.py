from pymongo import MongoClient
from bson.objectid import ObjectId

my_id = ObjectId('638e20916d566a8dd483b26d')
print(my_id)
cluster = "mongodb://lebpc20.epfl.ch/2701"

client = MongoClient(cluster)

for db in client.list_database_names():
    print(db)
print(client)

db = client.deep_events

import datetime

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "isim"],
        "date": datetime.datetime.utcnow()}

post2 = {"author": "Juan",
        "text": "My second blog post!",
        "tags": ["mongodb", "java", "isim"],
        "date": datetime.datetime.utcnow()}

post3 = {"_id": my_id,
        "author": "Other",
        "text": "My third blog post!",
        "tags": ["mongodb", "python", "Zeiss"],
        "date": datetime.datetime.utcnow()}

coll = db['collection']
coll.insert_one(post3).inserted_id

for item in coll.find():
    print(item)

# print(post_id)
# 7eea 7d48 273f 4f08 91db 7295 8dd0 6fc0