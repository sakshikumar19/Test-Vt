import streamlit as st
import pandas as pd
import os
import time
import json
import yaml
from datetime import datetime
import subprocess
from typing import List, Dict, Any, Optional
from chatbot import VitessFAQChatbot, initialize_inference_llm
import pandas as pd
import asyncio
import sys

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
        
        with open(self.text_log_file, 'w',encoding="utf-8") as file:
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
        with open(self.text_log_file, 'a', encoding="utf-8") as file:
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
        
        with open(detailed_log_file, 'w', encoding="utf-8") as file:
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
        
        # Return a deep copy of the result to avoid any reference issues
        return {
            "result": {
                "answer": result.get("answer", ""),
                "documents": retrieved_docs
            },
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
        
        with open(batch_log_file, 'w',encoding="utf-8") as file:
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
            with open(batch_log_file, 'a',encoding="utf-8") as file:
                file.write(f"QUESTION {i+1}: {question}\n")
                file.write(f"Answer: {eval_result['result']['answer'][:100]}...\n")
                file.write(f"Docs Retrieved: {eval_result['evaluation']['num_docs_retrieved']}\n")
                file.write(f"Top Score: {eval_result['evaluation']['top_doc_score']}\n")
                file.write(f"Response Time: {eval_result['evaluation']['response_time_ms']} ms\n")
                file.write(f"LLM Judge: {result_dict.get('llm_judge', 'Not evaluated')}\n")
                file.write("-" * 50 + "\n\n")
        
        # Write summary at the end
        with open(batch_log_file, 'a',encoding="utf-8") as file:
            file.write("=== SUMMARY STATISTICS ===\n")
            avg_docs = sum(r["num_docs"] for r in results) / len(results) if results else 0
            avg_score = sum(r["top_score"] for r in results if r["top_score"] is not None) / len([r for r in results if r["top_score"] is not None]) if [r for r in results if r["top_score"] is not None] else 0
            avg_time = sum(r["response_time_ms"] for r in results) / len(results) if results else 0
            
            file.write(f"Average Documents Retrieved: {avg_docs:.2f}\n")
            file.write(f"Average Top Score: {avg_score:.3f}\n")
            file.write(f"Average Response Time: {avg_time:.0f} ms\n")
        
        # Still return a DataFrame for display in the Streamlit app
        return pd.DataFrame(results)
    
    def evaluate_with_ground_truth(self, yaml_path: str) -> pd.DataFrame:
        """Evaluate chatbot using YAML file with ground truth expected chunks"""
        results = []
        
        # Load YAML file
        try:
            with open(yaml_path, 'r', encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
        except Exception as e:
            return pd.DataFrame(), {"error": str(e)}, []
        
        test_cases = yaml_data.get('test_cases', [])
        total_chunks_found = 0
        total_ground_truth_chunks = 0
        
        # Create a special ground truth log file
        ground_truth_log_file = os.path.join(
            self.log_path, 
            f"ground_truth_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # Open log file once for the entire function
        with open(ground_truth_log_file, 'w', encoding="utf-8") as log_file:
            log_file.write("=== GROUND TRUTH EVALUATION RESULTS ===\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Test Cases: {len(test_cases)}\n")
            log_file.write("=" * 50 + "\n\n")
            
            for i, test_case in enumerate(test_cases):
                question = test_case.get('question', '')
                expected_chunks = test_case.get('expected_chunks', [])
                
                # Skip if no question or expected chunks
                if not question or not expected_chunks:
                    continue
                
                # Evaluate query
                try:
                    eval_result = self.evaluate_query(question)
                    retrieved_docs = eval_result["result"]["documents"]
                    retrieved_chunks = [doc.content for doc in retrieved_docs]
                    retrieved_chunks_text = "\n".join(retrieved_chunks)
                    
                    # Check for expected chunks in retrieved docs
                    chunks_found = 0
                    found_chunks = []
                    missing_chunks = []
                    
                    for expected_chunk in expected_chunks:
                        # Check if expected chunk is a substring of any retrieved chunk
                        chunk_found = False
                        for retrieved_chunk in retrieved_chunks:
                            if expected_chunk in retrieved_chunk:
                                chunk_found = True
                                found_chunks.append(expected_chunk)
                                break
                        
                        if not chunk_found:
                            missing_chunks.append(expected_chunk)
                        else:
                            chunks_found += 1
                    
                    # Calculate accuracy for this question
                    total_expected = len(expected_chunks)
                    accuracy = (chunks_found / total_expected * 100) if total_expected > 0 else 0
                    
                    # Update totals for overall accuracy
                    total_ground_truth_chunks += total_expected
                    total_chunks_found += chunks_found
                    
                    # Build result dictionary
                    result_dict = {
                        "question": question,
                        "expected_chunks": expected_chunks,
                        "retrieved_chunks": retrieved_chunks,
                        "answer": eval_result["result"]["answer"],
                        "chunks_found": chunks_found,
                        "total_expected": total_expected,
                        "accuracy": accuracy,
                        "found_chunks": found_chunks,
                        "missing_chunks": missing_chunks,
                        "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                        "top_score": eval_result["evaluation"]["top_doc_score"],
                        "response_time_ms": eval_result["evaluation"]["response_time_ms"]
                    }
                    
                    # LLM Judge evaluation commented out as it wasn't being used
                    result_dict["llm_judge"] = "Not evaluated"
                    
                    results.append(result_dict)
                    
                    # Log the result
                    log_file.write(f"TEST CASE {i+1}: {question}\n")
                    log_file.write(f"Expected Chunks: {len(expected_chunks)}\n")
                    for j, chunk in enumerate(expected_chunks):
                        log_file.write(f"  {j+1}. {chunk}\n")
                    log_file.write(f"Retrieved Chunks: {len(retrieved_chunks)}\n")
                    for j, chunk in enumerate(retrieved_chunks):
                        log_file.write(f"  {j+1}. {chunk[:100]}...\n")
                    log_file.write(f"Chunks Found: {chunks_found}/{total_expected}\n")
                    log_file.write(f"Accuracy: {accuracy:.2f}%\n")
                    log_file.write(f"Missing Chunks: {missing_chunks}\n")
                    log_file.write(f"Answer: {eval_result['result']['answer'][:200]}...\n")
                    log_file.write(f"LLM Judge: {result_dict.get('llm_judge', 'Not evaluated')}\n")
                    log_file.write("-" * 50 + "\n\n")
                except Exception as e:
                    log_file.write(f"ERROR processing test case {i+1}: {str(e)}\n")
                    log_file.write("-" * 50 + "\n\n")
                    continue
            
            # Calculate overall accuracy
            overall_accuracy = (total_chunks_found / total_ground_truth_chunks * 100) if total_ground_truth_chunks > 0 else 0
            
            # Log summary
            log_file.write("=== SUMMARY STATISTICS ===\n")
            log_file.write(f"Total Test Cases: {len(test_cases)}\n")
            log_file.write(f"Total Expected Chunks: {total_ground_truth_chunks}\n")
            log_file.write(f"Total Chunks Found: {total_chunks_found}\n")
            log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        
        # Create summary
        summary = {
            "total_test_cases": len(test_cases),
            "total_expected_chunks": total_ground_truth_chunks,
            "total_chunks_found": total_chunks_found,
            "overall_accuracy": overall_accuracy
        }
        
        # Prepare simplified DataFrame for display and CSV export
        display_results = []
        for result in results:
            display_result = {
                "question": result["question"],
                "answer": result["answer"],
                "expected_chunks": "\n".join(result["expected_chunks"]),
                "retrieved_chunks": "\n".join(result["retrieved_chunks"]),
                "accuracy": f"{result['accuracy']:.2f}%",
                "found_chunks": result["chunks_found"],
                "total_expected": result["total_expected"],
                "llm_judge": result.get("llm_judge", "Not evaluated")
            }
            display_results.append(display_result)
        
        return pd.DataFrame(display_results), summary, results

def run_extract_docs():
    """Run the extract_docs.py script to clone repos and extract data"""
    try:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extract_docs.py"))

        # Check if the script exists
        if os.path.exists(script_path):
            try:
                result = subprocess.run([sys.executable, script_path], 
                                        check=True, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
                print("Script output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script_path}: {e.stderr}")
        else:
            print(f"Error: {script_path} not found. Ensure the file exists in the correct location.")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error executing extract_docs.py: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"

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
        
        # Add chunk size and overlap parameters to sidebar
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=400, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
        
        # Retrieval settings
        top_k = st.slider("Top K Documents", min_value=1, max_value=10, value=3)
        
        document_paths = [
            "./data/troubleshoot.txt",
            "./data/design-docs.txt",
            "./data/faq.txt",
            "./data/flags.txt",
            "./data/v22.txt"
        ]
    # Create tabs for different evaluation modes - removed the last two tabs
    tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Evaluation", "Ground Truth Evaluation"])
    
    # Check if we need to run extract_docs.py first
    if "extract_docs_run" not in st.session_state:
        with st.spinner("Running on-demand cloning and data collection..."):
            success, message = run_extract_docs()
            if success:
                st.success("Data collection completed successfully!")
            else:
                pass
                # st.error(f"Error during data collection: {message}")
            st.session_state.extract_docs_run = True
    
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
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )

                # # In-memory document store
                # chatbot = VitessFAQChatbot(
                #     embedding_model="sentence-transformers/all-mpnet-base-v2",
                #     llm_model=llm,
                #     document_store_type="inmemory"
                # )

                # # FAISS document store
                # chatbot = VitessFAQChatbot(
                #     embedding_model="sentence-transformers/all-mpnet-base-v2",
                #     llm_model=llm,
                #     document_store_type="chroma",
                #     document_store_type="chroma",
                #     collection_name="my_vitess_collection",
                #     persist_directory="./"
                # )
                
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
                
                # LLM Judge is commented out as it wasn't being used
                llm_judgment = "Not evaluated"
                
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
                        st.text_area(f"Content Doc {i+1}", doc.content, height=200, key=f"single_query_doc_{i}")
                
                # Add to session history
                st.session_state.eval_results.append({
                    "question": question,
                    "answer": result["answer"],
                    "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                    "top_score": eval_result["evaluation"]["top_doc_score"],
                    "response_time_ms": eval_result["evaluation"]["response_time_ms"],
                    "llm_judge": llm_judgment
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
    
    # Ground Truth Evaluation Tab
    with tab3:
        st.header("Ground Truth Evaluation")
        
        # File uploader for YAML file
        uploaded_file = st.file_uploader("Upload YAML file with test cases and expected chunks", type=["yaml", "yml"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Or provide a path to a YAML file
            yaml_path = st.text_input("Or enter path to YAML file", "test_cases.yaml")
        
        with col2:
            # Example YAML structure
            st.markdown("### Expected YAML structure:")
            st.code("""
test_cases:
  - question: "What versions of MySQL are compatible?"
    expected_chunks:
      - "refer to our Supported Databases"
  - question: "Another question here"
    expected_chunks:
      - "Expected chunk 1"
      - "Expected chunk 2"
            """)
        
        # Button to run evaluation
        run_button = st.button("Run Ground Truth Evaluation")
        
        if run_button and (uploaded_file is not None or os.path.exists(yaml_path)):
            with st.spinner("Evaluating against ground truth..."):
                # If uploaded file exists, save it temporarily
                if uploaded_file is not None:
                    # Save uploaded file to a temporary file
                    temp_yaml_path = f"temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                    with open(temp_yaml_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    yaml_path = temp_yaml_path
                
                # Run evaluation
                display_df, summary, detailed_results = evaluator.evaluate_with_ground_truth(yaml_path)
                
                # Store in session state
                st.session_state.ground_truth_results = detailed_results
                st.session_state.ground_truth_display = display_df
                st.session_state.ground_truth_summary = summary
                
                # Delete temp file if it was created
                if uploaded_file is not None and os.path.exists(temp_yaml_path):
                    os.remove(temp_yaml_path)
        
        # Show results if available
        if "ground_truth_summary" in st.session_state:
            summary = st.session_state.ground_truth_summary
            
            # Display overall metrics
            st.subheader("Overall Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Test Cases", summary["total_test_cases"])
            col2.metric("Chunks Found", f"{summary['total_chunks_found']}/{summary['total_expected_chunks']}")
            col3.metric("Overall Accuracy", f"{summary['overall_accuracy']:.2f}%")
            
            # Display detailed results
            st.subheader("Detailed Results")
            st.dataframe(st.session_state.ground_truth_display)
            
            # Prepare CSV for download (more detailed version)
            csv_data = st.session_state.ground_truth_display.to_csv(index=False)
            
            st.download_button(
                "Download Ground Truth Results CSV",
                csv_data,
                "ground_truth_evaluation.csv",
                "text/csv"
            )
            
            # Show per-question details with expanders
            st.subheader("Per-Question Analysis")
            
            for result in st.session_state.ground_truth_results:
                accuracy_text = f"{result['accuracy']:.2f}% ({result['chunks_found']}/{result['total_expected']} chunks)"
                llm_judgment = result.get("llm_judge", "Not evaluated")
                with st.expander(f"Q: {result['question']} - Accuracy: {accuracy_text} - LLM: {llm_judgment}"):
                    st.markdown("#### Generated Answer")
                    st.write(result['answer'])
                    
                    st.markdown("#### Ground Truth Chunks")
                    for i, chunk in enumerate(result['expected_chunks']):
                        found = chunk in result['found_chunks']
                        status = "[FOUND]" if found else "[MISSING]"
                        st.markdown(f"{status} {chunk}")
                    
                    st.markdown("#### Retrieved Chunks")
                    for i, chunk in enumerate(result['retrieved_chunks']):
                        st.text_area(
                            f"Chunk {result['question'][:10]}_{i+1}", 
                            chunk, 
                            height=100,
                            key=f"ground_truth_chunk_{hash(result['question'])}_{i}"
                        )

if __name__ == "__main__":
    run_streamlit_eval()