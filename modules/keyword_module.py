import yake

class KeywordsModule:
    def __init__(self, top_k=15):
        self.top_k = top_k
        self.kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,                 
            dedupLim=0.8,         
            top=self.top_k * 3,   
            features=None
        )

    def extract_keywords(self, text):
        if not text or not text.strip():
            return []

        keywords = self.kw_extractor.extract_keywords(text)

        keywords = sorted(keywords, key=lambda x: x[1])
        results = [kw for kw, score in keywords]

        seen = set()
        final = []
        for kw in results:
            key = kw.lower()
            if key not in seen:
                seen.add(key)
                final.append(kw)
            if len(final) >= self.top_k:
                break
        return final

