import uuid
import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="research_memory"
)


def add_documents(documents):

    for doc in documents:

        collection.add(
            documents=[doc],
            ids=[str(uuid.uuid4())]
        )


def retrieve(query):

    count = collection.count()

    if count == 0:
        return {"documents": [["No context available yet."]]}

    results = collection.query(
        query_texts=[query],
        n_results=min(3, count)  
    )

    return results
