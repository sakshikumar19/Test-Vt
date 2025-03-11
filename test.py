from chatbot import initialize_inference_llm
import time
import re
from typing import List

llm = initialize_inference_llm()

class LocalLLMAnswerGenerator:
    """Generates answers using a pre-loaded language model"""
    def __init__(self, local_llm):
        self.llm = local_llm
        print("Using pre-loaded language model")

    def generate(self, context: str, question: str, history: List = None) -> str:
        """Generate an answer based on context and question"""
        print("Starting answer generation...")

        history_text = ""
        if history:
            print(f"Including {len(history)} previous exchanges in prompt")
            for q, a in history[-2:]:  # Last 2 exchanges
                history_text += f"Human: {q}\nAssistant: {a}\n\n"

        # Add context truncation
        max_context_length = 3000  # Safe limit to prevent exceeding token limit
        if len(context) > max_context_length:
            print(f"Context too long ({len(context)} chars), truncating to ~{max_context_length} chars")
            # Split context into sentences and rebuild up to the limit
            sentences = re.split(r'(?<=[.!?])\s+', context)
            truncated_context = ""
            for sentence in sentences:
                if len(truncated_context) + len(sentence) + 1 <= max_context_length:
                    truncated_context += sentence + " "
                else:
                    break
            context = truncated_context.strip()
            print(f"Truncated context length: {len(context)} chars")

        prompt = f"""<s>[INST] You are an expert Vitess support assistant. Vitess is an open-source database clustering system for horizontal scaling of MySQL.

Answer the question based ONLY on the following context. If you don't know the answer or the information is not in the context, say "I don't have enough information to answer this question" and suggest what information might help.

Be accurate, helpful, concise, and clear. Format your answer using markdown when appropriate.

Context:
{context}

{history_text}
Question: {question} [/INST]
"""

        try:
            print(f"Context length: {len(context)} characters")
            print(f"Prompt length: {len(prompt)} characters")
            print("Starting LLM inference...")

            approx_tokens = len(prompt.split())
            print(f"Approximate token count: {approx_tokens}")

            inference_start = time.time()
            print("Sending request to model...")

            answer = self.llm(prompt)
            print(f"Model response received in {time.time() - inference_start:.2f}s")

            if isinstance(answer, str):
                # Direct string output
                response = answer
            elif isinstance(answer, list) and len(answer) > 0:
                # For models that return a list of responses
                response = answer[0]
                # Check if response has a text attribute
                if hasattr(response, 'text'):
                    response = response.text
                elif isinstance(response, dict) and 'generated_text' in response:
                    response = response['generated_text']
                else:
                    response = str(response)
            else:
                response = str(answer)

            # Clean up response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            response = response.strip('"\'')

            print(f"Processed answer (length: {len(response)} chars)")

            # fallback message
            if not response.strip():
                print("Empty answer after extraction, using fallback")
                response = "Based on the available information, I couldn't find specific details about this query in Vitess. Please check the official Vitess documentation for more information."

            return response
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
        return "I apologize, but I encountered an error generating a response. Please try rephrasing your question or ask about a different Vitess topic."

answer_generator = LocalLLMAnswerGenerator(llm)

answer = answer_generator.generate(
    context="i love animals",
    question="do i love birds?",
    history=[]
)


print(answer)