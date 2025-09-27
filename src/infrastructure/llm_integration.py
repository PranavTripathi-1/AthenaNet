from transformers import pipeline

class LLMExplainer:
    def __init__(self, model_name="distilgpt2"):
        self.generator = pipeline("text-generation", model=model_name)

    def explain_prediction(self, text_input: str, prediction: str):
        prompt = f"The system predicts '{prediction}' risk. Explain this in simple words for the user.\nUser text: {text_input}\nExplanation:"
        response = self.generator(prompt, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']
