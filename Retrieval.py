
import Datasources

def triggerretrieval(collection: str, prompt:str):

    vector = Datasources.ChromaDB(collection)
    results = vector.semanticRetrieval(prompt, n_results=5)
    return results
