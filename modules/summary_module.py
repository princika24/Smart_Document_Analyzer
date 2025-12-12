import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

nltk.download("punkt", quiet=True)


class SummaryModule:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.paraphraser = pipeline(
            "text2text-generation",
            model="t5-small",
            tokenizer="t5-small",
            device=-1  
        )

    def _clean_text(self, text: str) -> str:
        """Basic cleanup of raw text."""
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^A-Za-z0-9.,;:'\"()\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _filter_sentences(self, text: str):
        """Remove irrelevant, quiz-like, or broken lines."""
        sentences = sent_tokenize(text)
        filtered = []

        for s in sentences:
            s = s.strip()

            s = re.sub(
                r"^[^A-Za-z]*?(true|false|paraphrase|question|alse|rue|araphrase)\b[:\-]?\s*",
                "",
                s,
                flags=re.IGNORECASE,
            )

            if not s or len(s.split()) < 6:
                continue

            if "?" in s or s.lower().startswith(
                ("choose", "tick", "mark", "select", "what is")
            ):
                continue

            filtered.append(s)

        return filtered

    def _rank_sentences(self, sentences, top_k):
        """Select the most relevant sentences using embeddings."""
        if len(sentences) <= top_k:
            return sentences

        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        doc_emb = embeddings.mean(dim=0, keepdim=True)
        sims = util.cos_sim(doc_emb, embeddings)[0]
        top_idx = sims.argsort(descending=True)[:top_k]

        ranked = [sentences[i] for i in top_idx]

        order = {s: i for i, s in enumerate(sentences)}
        ranked.sort(key=lambda s: order[s])
        return ranked

    def _paraphrase(self, sentence: str) -> str:
        """Paraphrase a sentence lightly to improve fluency."""
        try:
            result = self.paraphraser(
                f"paraphrase: {sentence}",
                max_length=min(80, len(sentence.split()) + 25),
                num_beams=4,
                do_sample=False,
            )[0]["generated_text"]
            result = re.sub(r"(?i)paraphrase\s*:\s*", "", result).strip()
            return result
        except Exception:
            return sentence

    def _deduplicate_summary(self, text: str) -> str:
        """Remove near-duplicate or repeated sentences in the final summary."""
        seen = set()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned = []

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            key = re.sub(r"\s+", " ", s_clean.lower())

            if key in seen:
                continue
            seen.add(key)

            if re.match(r"^(instead of following|in this approach we teach|false)", key):
                continue

            cleaned.append(s_clean)

        return " ".join(cleaned)

    def _merge_sentences(self, sentences):
        """Combine ranked/paraphrased sentences into coherent paragraphs."""
        paras, cur = [], []

        for s in sentences:
            cur.append(s)
            if len(" ".join(cur).split()) > 80:
                paras.append(" ".join(cur))
                cur = []

        if cur:
            paras.append(" ".join(cur))

        return "\n\n".join(paras)

    def generate_summary(self, text: str, level="medium") -> str:
        """Main public function to generate summary."""
        text = self._clean_text(text)
        if not text:
            return "No text provided for summarization."

        sentences = self._filter_sentences(text)
        if not sentences:
            return "Summary not available."

        if level == "short":
            num_sentences = 4
        elif level == "medium":
            num_sentences = 8
        else:
            num_sentences = 14

        ranked = self._rank_sentences(sentences, num_sentences)
        paraphrased = [self._paraphrase(s) for s in ranked]

        final_summary = self._merge_sentences(paraphrased)
        final_summary = self._deduplicate_summary(final_summary)

        return final_summary.strip()
