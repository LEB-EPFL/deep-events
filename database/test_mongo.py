from pymongo import MongoClient


cluster = "mongodb+srv://stepp:(HArtliebstr3)@lebpc20"

client = MongoClient(cluster)

for db in client.list_databases():
    print(db)
print(client)

db = client.test_database

import datetime

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "isim"],
        "date": datetime.datetime.utcnow()}

post2 = {"author": "Juan",
        "text": "My second blog post!",
        "tags": ["mongodb", "java", "isim"],
        "date": datetime.datetime.utcnow()}

post3 = {"author": "Santi",
        "text": "My third blog post!",
        "tags": ["mongodb", "python", "Zeiss"],
        "date": datetime.datetime.utcnow()}

coll = db.collection
post_id = coll.insert_one(post).inserted_id

print(post_id)
