import ollama

class OllamaGenerator:
    def __init__(self, model_name="qwen3:0.6b"):
        self.model_name = model_name

    def generate_answer(self, query: str, context: str, history: str = "", sources: str = ""):
        # Include history for conversation coherence
        history_lines = history.strip().split("\n") if history.strip() else []
        if len(history_lines) > 10:
            history_lines = history_lines[-10:]
        trimmed_history = "\n".join(history_lines)

        prompt = f"""You are a helpful, professional, and friendly Smart Legal Assistant.

Analyze the retrieved legal records and answer the question.

Your response MUST follow this exact structure and order. Use the headers exactly as shown. Do not repeat information or output fragmented sentences. Keep explanations concise, legally accurate, and easy to understand. Speak naturally like ChatGPT (use bullet points and bolding where appropriate, but do not dump raw text).

Ensure that Fine, Imprisonment, and Victim Compensation are kept as separate details (do not merge unrelated sentences, e.g. do not state that imprisonment must be paid to the victim).

IMPORTANT: Output exactly ONE instance of each header in your final response. Do NOT repeat the entire structure for different retrieved records. Synthesize all relevant records to answer the single user query under one unified set of headers. If a retrieved record is completely unrelated to the user's query, ignore it.

Structure:
### Definition
[Concise explanation of the crime in plain, natural English]

### Applicable IPC Section
[State the IPC section number(s) directly]

### Applicable BNS Section
[State the BNS section number(s) directly]

### Punishment
[State the statutory punishment clearly. Detail imprisonment, fine, and compensation as distinct details]

### Additional Notes
[List related sections, exceptions, or key notes if available; otherwise write "None"]

### Sources
{sources if sources else "None"}

Legal Records:
{context}

Conversation History:
{trimmed_history}

Question: {query}
Answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=True
            )
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            yield f"Ollama Error: {str(e)}"
