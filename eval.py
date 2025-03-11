import streamlit as st
import pandas as pd
import os
import time
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from chatbot import VitessFAQChatbot, initialize_inference_llm, initialize_llm
import pandas as pd
import asyncio

# On Windows, use this event loop policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class ChatbotEvaluator:
    def __init__(self, chatbot: VitessFAQChatbot, log_path: str = "evaluation_logs"):
        self.chatbot = chatbot
        self.log_path = log_path
        self.csv_log_file = os.path.join(log_path, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.detailed_logs_dir = os.path.join(log_path, "detailed_logs")
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.detailed_logs_dir, exist_ok=True)
        
        # Initialize CSV log file with headers
        with open(self.csv_log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "timestamp", "question", "answer", "num_docs", 
                "top_doc_score", "response_time_ms", "doc1_source", "doc1_score", 
                "doc2_source", "doc2_score", "doc3_source", "doc3_score"
            ])
    
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
        
        # Prepare document data for CSV
        doc_data = []
        for i in range(3):  # Capture up to 3 docs
            if i < len(retrieved_docs):
                doc = retrieved_docs[i]
                doc_data.extend([
                    doc.meta.get("source", "Unknown"),
                    doc.score if hasattr(doc, "score") else None
                ])
            else:
                doc_data.extend(["", None])  # Empty placeholders
        
        # Log to CSV file
        with open(self.csv_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                question,
                result.get("answer", "")[:500],  # Truncate long answers
                num_docs,
                top_doc_score,
                round(total_time_ms, 2),
                *doc_data  # Unpack document data
            ])
        
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
                "top_doc_score": top_doc_score,
            }
        }

    def batch_evaluate(self, test_questions: List[str]) -> pd.DataFrame:
        """Run evaluation on a batch of test questions"""
        results = []
        
        # Create a special batch log file (CSV)
        batch_log_file = os.path.join(
            self.log_path, 
            f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        # Initialize batch CSV with headers
        with open(batch_log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "question_id", "question", "answer", "num_docs", 
                "top_score", "response_time_ms", "doc1_source", "doc2_source", "doc3_source"
            ])
        
        for i, question in enumerate(test_questions):
            eval_result = self.evaluate_query(question)
            
            # Prepare documents info
            docs = eval_result["result"].get("documents", [])
            doc_sources = [doc.meta.get("source", "Unknown") for doc in docs]
            # Pad with empty strings if less than 3 documents
            doc_sources = doc_sources + [""] * (3 - len(doc_sources)) if len(doc_sources) < 3 else doc_sources[:3]
            
            result_dict = {
                "question": question,
                "answer": eval_result["result"]["answer"],
                "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                "top_score": eval_result["evaluation"]["top_doc_score"],
                "response_time_ms": eval_result["evaluation"]["response_time_ms"],
                "doc_sources": doc_sources
            }
            results.append(result_dict)
            
            # Append to the batch CSV
            with open(batch_log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    i+1,
                    question,
                    eval_result['result']['answer'][:200],  # Truncate long answers
                    eval_result['evaluation']['num_docs_retrieved'],
                    eval_result['evaluation']['top_doc_score'],
                    eval_result['evaluation']['response_time_ms'],
                    *doc_sources  # Unpack document sources
                ])
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Write summary statistics to a separate CSV
        summary_file = os.path.join(self.log_path, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_questions", len(results)])
            writer.writerow(["avg_docs_retrieved", f"{sum(r['num_docs'] for r in results) / len(results):.2f}"])
            writer.writerow(["avg_top_score", f"{sum(r['top_score'] for r in results if r['top_score'] is not None) / len(results):.3f}"])
            writer.writerow(["avg_response_time_ms", f"{sum(r['response_time_ms'] for r in results) / len(results):.0f}"])
        
        return df


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
        
        # LLM model type (local or Groq)
        llm_type = st.radio(
            "LLM Type", 
            ["Local", "Groq API"],
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
                    llm = initialize_llm(llm_model)
                else:
                    llm = initialize_inference_llm(model_name=llm_model)
                
                chatbot = VitessFAQChatbot(
                    embedding_model=embedding_model,
                    llm_model=llm,
                    top_k=top_k,
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
                col1, col2, col3, col4 = st.columns(4)
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
                col1, col2, col3, col4 = st.columns(4)
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
                    if f.endswith('.csv') or f.endswith('.log')]
        
        if log_files:
            selected_log = st.selectbox("Select log file to view:", log_files)
            log_path = os.path.join(evaluator.log_path, selected_log)
            
            if os.path.exists(log_path):
                if log_path.endswith('.csv'):
                    # Display CSV as table
                    df = pd.read_csv(log_path)
                    st.dataframe(df)
                else:
                    # Display text log
                    with open(log_path, 'r') as file:
                        log_content = file.read()
                    st.text_area("Log Content", log_content, height=500)
                
                # Download option
                if log_path.endswith('.csv'):
                    with open(log_path, 'r') as file:
                        csv_content = file.read()
                    st.download_button(
                        "Download CSV File",
                        csv_content,
                        selected_log,
                        "text/csv"
                    )
                else:
                    with open(log_path, 'r') as file:
                        log_content = file.read()
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
            col1, col2, col3, col4 = st.columns(4)
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
