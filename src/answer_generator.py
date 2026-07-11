from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates an answer based ONLY on the provided context.
        """
        prompt = (
            "Based on the following legal context, provide a detailed and complete explanation in natural sentences. "
            "Make sure to explain the crime, its description, the punishment, and any relevant sections. "
            "Do not output isolated numbers or short phrases. Write a comprehensive response.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Detailed Explanation:"
        )
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=30,
                max_new_tokens=250,
                temperature=0.5,
                repetition_penalty=1.2,
                do_sample=True,
                num_beams=4,
                early_stopping=True
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not answer:
                return "I couldn't find sufficient information in the uploaded legal documents."
            return str(answer)
        except Exception as e:
            import logging
            logger = logging.getLogger("rag_engine")
            logger.exception(e)
            return f"Error: {str(e)}"
