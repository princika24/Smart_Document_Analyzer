from sentence_transformers import SentenceTransformer, util

class DocumentRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.sentences = []
        self.embeddings = None

    def index(self, text, chunk_size=400):
        self.sentences = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.embeddings = self.model.encode(self.sentences, convert_to_tensor=True)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        return [(self.sentences[h['corpus_id']], h['score']) for h in hits]
