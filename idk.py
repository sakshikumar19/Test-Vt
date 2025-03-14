from chatbot2 import initialize_inference_llm

llm = initialize_inference_llm()

prompt = f"""<s>[INST] You are an expert answer checker.

You will be given a question and an answer. Your task is to evaluate whether the answer accurately and completely responds to the question. 

Reply with "Yes" if the answer is correct and fully addresses all parts of the question.  
Reply with "No" if the answer is incorrect, incomplete, or contains inaccuracies.  

Question: {q}

Answer: {a}

Your Response:"""

answer = llm(prompt)
print(answer)