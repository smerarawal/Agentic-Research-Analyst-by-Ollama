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

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    return results
