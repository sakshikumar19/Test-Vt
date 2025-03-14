import streamlit as st
import pandas as pd
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any
from chatbot1 import VitessFAQChatbot, initialize_inference_llm
import pandas as pd
import asyncio

# On Windows, use this event loop policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class ChatbotEvaluator:
    def __init__(self, chatbot: VitessFAQChatbot, log_path: str = "evaluation_logs"):
        self.chatbot = chatbot
        self.log_path = log_path
        self.text_log_file = os.path.join(log_path, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.detailed_logs_dir = os.path.join(log_path, "detailed_logs")
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.detailed_logs_dir, exist_ok=True)
        
        with open(self.text_log_file, 'w') as file:
            file.write("=== VITESS FAQ CHATBOT EVALUATION LOG ===\n")
            file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 50 + "\n\n")
    
    def evaluate_query(self, question: str) -> Dict[str, Any]:
        """Evaluate a query and log the results"""
        start_time = time.time()
        
        # Run query
        result = self.chatbot.query(question)
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        
        # Extract documents
        retrieved_docs = result.get("documents", [])
        num_docs = len(retrieved_docs)
        
        # Get top document score if available
        top_doc_score = retrieved_docs[0].score if num_docs > 0 and hasattr(retrieved_docs[0], "score") else None
        
        # Extract document content and sources
        chunks = [doc.content for doc in retrieved_docs]
        sources = [doc.meta.get("source", "Unknown") for doc in retrieved_docs]
        
        # Log to the main text log file
        with open(self.text_log_file, 'a') as file:
            file.write(f"QUERY: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Docs Retrieved: {num_docs}\n")
            file.write(f"Top Doc Score: {top_doc_score}\n")
            file.write(f"Response Time: {round(total_time_ms, 2)} ms\n")
            file.write(f"Sources: {', '.join(sources[:3])}\n")
            file.write("-" * 50 + "\n\n")
        
        # Write detailed log to separate text file
        detailed_log_file = os.path.join(
            self.detailed_logs_dir, 
            f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        with open(detailed_log_file, 'w') as file:
            file.write(f"DETAILED QUERY LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 70 + "\n\n")
            file.write(f"QUESTION: {question}\n\n")
            file.write(f"ANSWER: {result.get('answer', '')}\n\n")
            file.write(f"METRICS:\n")
            file.write(f"  - Documents Retrieved: {num_docs}\n")
            file.write(f"  - Top Document Score: {top_doc_score}\n")
            file.write(f"  - Response Time: {round(total_time_ms, 2)} ms\n")
            file.write(f"  - Retrieval Time (est.): {round(total_time_ms * 0.7, 2)} ms\n\n")
            
            file.write("RETRIEVED DOCUMENTS:\n")
            for i, doc in enumerate(retrieved_docs):
                file.write(f"Document {i+1}:\n")
                file.write(f"  Source: {doc.meta.get('source', 'Unknown')}\n")
                file.write(f"  Score: {doc.score if hasattr(doc, 'score') else 'N/A'}\n")
                file.write(f"  Content:\n{'-' * 40}\n{doc.content}\n{'-' * 40}\n\n")
        
        return {
            "result": result,
            "evaluation": {
                "response_time_ms": round(total_time_ms, 2),
                "num_docs_retrieved": num_docs,
                "top_doc_score": top_doc_score
            }
        }

    def batch_evaluate(self, test_questions: List[str]) -> pd.DataFrame:
        """Run evaluation on a batch of test questions"""
        results = []
        
        # Create a special batch log file
        batch_log_file = os.path.join(
            self.log_path, 
            f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        with open(batch_log_file, 'w') as file:
            file.write("=== BATCH EVALUATION RESULTS ===\n")
            file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Questions: {len(test_questions)}\n")
            file.write("=" * 50 + "\n\n")
        
        for i, question in enumerate(test_questions):
            eval_result = self.evaluate_query(question)
            
            result_dict = {
                "question": question,
                "answer": eval_result["result"]["answer"],
                "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                "top_score": eval_result["evaluation"]["top_doc_score"],
                "response_time_ms": eval_result["evaluation"]["response_time_ms"]
            }
            results.append(result_dict)
            
            # Append to the batch log
            with open(batch_log_file, 'a') as file:
                file.write(f"QUESTION {i+1}: {question}\n")
                file.write(f"Answer: {eval_result['result']['answer'][:100]}...\n")
                file.write(f"Docs Retrieved: {eval_result['evaluation']['num_docs_retrieved']}\n")
                file.write(f"Top Score: {eval_result['evaluation']['top_doc_score']}\n")
                file.write(f"Response Time: {eval_result['evaluation']['response_time_ms']} ms\n")
                file.write("-" * 50 + "\n\n")
        
        # Write summary at the end
        with open(batch_log_file, 'a') as file:
            file.write("=== SUMMARY STATISTICS ===\n")
            avg_docs = sum(r["num_docs"] for r in results) / len(results)
            avg_score = sum(r["top_score"] for r in results if r["top_score"] is not None) / len(results)
            avg_time = sum(r["response_time_ms"] for r in results) / len(results)
            
            file.write(f"Average Documents Retrieved: {avg_docs:.2f}\n")
            file.write(f"Average Top Score: {avg_score:.3f}\n")
            file.write(f"Average Response Time: {avg_time:.0f} ms\n")
        
        # Still return a DataFrame for display in the Streamlit app
        return pd.DataFrame(results)

def run_streamlit_eval():
    st.set_page_config(page_title="Vitess FAQ Chatbot Evaluator", layout="wide")
    
    st.title("Vitess FAQ Chatbot Evaluation Pipeline")
    
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            index=0
        )
        
        llm_type = st.radio(
            "LLM Type", 
            ["Local (currently defaults to Groq)", "Groq API"],
            index=0
        )
        
        # Show appropriate model selection based on type
        if llm_type == "Local":
            llm_model = st.selectbox(
                "Local LLM Model",
                ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/phi-2"],
                index=0
            )
        else:  # Groq API
            llm_model = st.selectbox(
                "Groq API Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b", "mixtral-8x7b-32768", "gemma-7b-it"],
                index=0
            )
        
        # Retrieval settings
        top_k = st.slider("Top K Documents", min_value=1, max_value=10, value=3)
        
        # File paths for ingestion
        st.subheader("Document Ingestion")
        document_paths = st.text_area(
            "Document Paths (one per line)",
            "./data/troubleshoot.txt\n./data/design-docs.txt\n./data/faq.txt\n./data/flags.txt\n./data/v22.txt"
        ).strip().split("\n")
    
    # Create tabs for different evaluation modes
    tab1, tab2, tab3, tab4 = st.tabs(["Single Query", "Batch Evaluation", "View Logs", "Test Cases"])
    
    # Initialize chatbot if not in session state
    if "chatbot" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            try:
                # Initialize LLM based on selection
                if llm_type == "Local":
                    llm = initialize_inference_llm()
                else:
                    llm = initialize_inference_llm()
                
                chatbot = VitessFAQChatbot(
                    embedding_model=embedding_model,
                    llm_model=llm,
                    chunk_size=300,
                    chunk_overlap=50,
                    top_k=top_k,
                    clone_repos=True,
                    cleanup_repos=True
                )
                
                # Ingest documents
                valid_paths = [path for path in document_paths if os.path.exists(path)]
                if valid_paths:
                    chatbot.ingest_documents(valid_paths)
                    st.sidebar.success(f"Ingested {len(valid_paths)} documents")
                else:
                    st.sidebar.warning("No valid document paths found")
                
                # Create evaluator
                evaluator = ChatbotEvaluator(chatbot)
                
                # Store in session state
                st.session_state.chatbot = chatbot
                st.session_state.evaluator = evaluator
                st.session_state.eval_results = []
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                st.exception(e)  # Show detailed error information
                st.stop()
    
    # Get from session state
    chatbot = st.session_state.chatbot
    evaluator = st.session_state.evaluator
    
    # Single Query Tab
    with tab1:
        st.header("Single Query Evaluation")
        
        question = st.text_input("Enter your question about Vitess:")
        
        if st.button("Submit Query") and question:
            with st.spinner("Processing query..."):
                eval_result = evaluator.evaluate_query(question)
                result = eval_result["result"]
                
                # Show evaluation metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Documents Retrieved", eval_result["evaluation"]["num_docs_retrieved"])
                col2.metric("Top Document Score", round(eval_result["evaluation"]["top_doc_score"], 3) if eval_result["evaluation"]["top_doc_score"] else "N/A")
                col3.metric("Response Time", f"{eval_result['evaluation']['response_time_ms']:.0f} ms")
                
                # Show answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Show retrieved documents with expanders
                st.subheader("Retrieved Documents")
                
                if not result["documents"]:
                    st.info("No documents retrieved")
                
                for i, doc in enumerate(result["documents"]):
                    score = doc.score if hasattr(doc, "score") else "N/A"
                    source = doc.meta.get("source", "Unknown")
                    
                    with st.expander(f"Document {i+1}: {source} (Score: {score})"):
                        st.text_area(f"Content {i+1}", doc.content, height=200)
                
                # Add to session history
                st.session_state.eval_results.append({
                    "question": question,
                    "answer": result["answer"],
                    "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                    "top_score": eval_result["evaluation"]["top_doc_score"],
                    "response_time_ms": eval_result["evaluation"]["response_time_ms"]
                })
    
    # Batch Evaluation Tab
    with tab2:
        st.header("Batch Evaluation")
        
        # Input area for test questions
        test_questions = st.text_area(
            "Enter test questions (one per line)",
            "What is Vitess?\nHow do I install Vitess?\nHow do I troubleshoot replication issues?\nWhat are the main flags for vtgate?"
        ).strip().split("\n")
        
        if st.button("Run Batch Evaluation") and test_questions:
            with st.spinner(f"Evaluating {len(test_questions)} questions..."):
                results_df = evaluator.batch_evaluate(test_questions)
                st.session_state.batch_results = results_df
                
                # Display results
                st.subheader("Batch Results")
                st.dataframe(results_df)
                
                # Summary metrics
                st.subheader("Summary Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg. Documents Retrieved", f"{results_df['num_docs'].mean():.2f}")
                col2.metric("Avg. Top Score", f"{results_df['top_score'].mean():.3f}")
                col3.metric("Avg. Response Time", f"{results_df['response_time_ms'].mean():.0f} ms")
                
                # Download results
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv_data,
                    "batch_results.csv",
                    "text/csv"
                )
    
    # View Logs Tab
    with tab3:
        st.header("Evaluation Logs")
        
        # List all log files
        log_files = [f for f in os.listdir(evaluator.log_path) 
                    if f.endswith('.txt') or f.endswith('.log')]
        
        if log_files:
            selected_log = st.selectbox("Select log file to view:", log_files)
            log_path = os.path.join(evaluator.log_path, selected_log)
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as file:
                    log_content = file.read()
                
                st.text_area("Log Content", log_content, height=500)
                
                # Download option
                st.download_button(
                    "Download Log File",
                    log_content,
                    selected_log,
                    "text/plain"
                )
            
            # Option to view detailed logs
            st.subheader("Detailed Query Logs")
            detailed_logs = [f for f in os.listdir(evaluator.detailed_logs_dir) 
                            if f.endswith('.log')]
            
            if detailed_logs:
                selected_detailed_log = st.selectbox("Select detailed log:", detailed_logs)
                detailed_log_path = os.path.join(evaluator.detailed_logs_dir, selected_detailed_log)
                
                if os.path.exists(detailed_log_path):
                    with open(detailed_log_path, 'r') as file:
                        detailed_content = file.read()
                    
                    st.text_area("Detailed Log Content", detailed_content, height=500)
            else:
                st.info("No detailed logs available yet.")
        else:
            st.info("No logs available yet. Run some evaluations first.")
            
    # Test Cases Tab
    with tab4:
        st.header("Manage Test Cases")
        
        # Load or initialize test cases
        test_cases_file = "test_cases.json"
        if "test_cases" not in st.session_state:
            if os.path.exists(test_cases_file):
                with open(test_cases_file, 'r') as f:
                    st.session_state.test_cases = json.load(f)
            else:
                st.session_state.test_cases = [
                    {"question": "What is Vitess?", "tags": ["basic", "introduction"]},
                    {"question": "How do I install Vitess?", "tags": ["installation"]},
                    {"question": "What flags does vtgate support?", "tags": ["configuration", "vtgate"]}
                ]
        
        # Add new test case
        st.subheader("Add New Test Case")
        new_question = st.text_input("New Test Question")
        new_tags = st.text_input("Tags (comma separated)")
        
        if st.button("Add Test Case") and new_question:
            tags = [tag.strip() for tag in new_tags.split(",")] if new_tags else []
            st.session_state.test_cases.append({"question": new_question, "tags": tags})
            with open(test_cases_file, 'w') as f:
                json.dump(st.session_state.test_cases, f)
            st.success("Test case added successfully!")
        
        # View and select test cases
        st.subheader("Available Test Cases")
        
        for i, case in enumerate(st.session_state.test_cases):
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.write(case["question"])
            col2.write(", ".join(case["tags"]))
            if col3.button("Run", key=f"run_{i}"):
                with st.spinner(f"Running test case: {case['question']}"):
                    eval_result = evaluator.evaluate_query(case["question"])
                    st.session_state.current_result = eval_result
                    st.experimental_rerun()
        
        # Display current result if available
        if "current_result" in st.session_state:
            st.subheader("Test Case Result")
            result = st.session_state.current_result["result"]
            
            # Show evaluation metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Documents Retrieved", st.session_state.current_result["evaluation"]["num_docs_retrieved"])
            col2.metric("Top Document Score", round(st.session_state.current_result["evaluation"]["top_doc_score"], 3) 
                        if st.session_state.current_result["evaluation"]["top_doc_score"] else "N/A")
            col3.metric("Response Time", f"{st.session_state.current_result['evaluation']['response_time_ms']:.0f} ms")
            
            # Show answer
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Show retrieved documents
            st.subheader("Retrieved Documents")
            for i, doc in enumerate(result["documents"]):
                score = doc.score if hasattr(doc, "score") else "N/A"
                source = doc.meta.get("source", "Unknown")
                
                with st.expander(f"Document {i+1}: {source} (Score: {score})"):
                    st.text_area(f"Content {i+1}", doc.content, height=200)


if __name__ == "__main__":
    run_streamlit_eval()
