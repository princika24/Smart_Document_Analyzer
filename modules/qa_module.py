from transformers import pipeline

class QAModule:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

    def answer(self, question, contexts, max_sentences=4):
        """
        Generate a detailed, coherent answer (3–5 sentences) from the top context.
        """
        merged_context = " ".join([ctx[0] for ctx in contexts[:3]])

        prompt = (
            f"Answer the following question in 3 to 5 full sentences based strictly on the provided document text.\n\n"
            f"Document text:\n{merged_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        result = self.generator(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.2
        )[0]["generated_text"]

        cleaned = result.strip()
        if cleaned.lower().startswith("answer:"):
            cleaned = cleaned[7:].strip()

        return cleaned
