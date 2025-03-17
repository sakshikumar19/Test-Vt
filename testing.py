import os
import time
import yaml
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

# Assuming chatbot2.py contains these imports - you'll need to adjust based on your actual implementation
from chatbot2 import VitessFAQChatbot, initialize_llm, initialize_inference_llm

# On Windows, use this event loop policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class SimpleChatbotEvaluator:
    def __init__(self, chatbot: VitessFAQChatbot, log_path: str = "evaluation_logs"):
        self.chatbot = chatbot
        self.log_path = log_path
        self.text_log_file = os.path.join(log_path, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.detailed_logs_dir = os.path.join(log_path, "detailed_logs")
        
        # Create log directories if they don't exist
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.detailed_logs_dir, exist_ok=True)
        
        # Initialize the main log file
        with open(self.text_log_file, 'w', encoding="utf-8") as file:
            file.write("=== VITESS FAQ CHATBOT EVALUATION LOG ===\n")
            file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 50 + "\n\n")
    
    def evaluate_query(self, question: str) -> Dict[str, Any]:
        """Evaluate a single query and log the results"""
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
            file.write(f"  - Response Time: {round(total_time_ms, 2)} ms\n\n")
            
            file.write("RETRIEVED DOCUMENTS:\n")
            for i, doc in enumerate(retrieved_docs):
                file.write(f"Document {i+1}:\n")
                file.write(f"  Source: {doc.meta.get('source', 'Unknown')}\n")
                file.write(f"  Score: {doc.score if hasattr(doc, 'score') else 'N/A'}\n")
                file.write(f"  Content:\n{'-' * 40}\n{doc.content}\n{'-' * 40}\n\n")
        
        # Return result dictionary
        return {
            "result": {
                "answer": result.get("answer", ""),
                "documents": retrieved_docs
            },
            "evaluation": {
                "response_time_ms": round(total_time_ms, 2),
                "num_docs_retrieved": num_docs,
                "top_doc_score": top_doc_score,
                "retrieved_chunks": chunks,
                "sources": sources
            }
        }
    
    def evaluate_with_ground_truth(self, yaml_path: str) -> tuple:
        """Evaluate chatbot using YAML file with ground truth expected chunks"""
        results = []
        
        # Load YAML file
        try:
            with open(yaml_path, 'r', encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading YAML file: {e}")
            return pd.DataFrame(), {"error": str(e)}, []
        
        test_cases = yaml_data.get('test_cases', [])
        total_chunks_found = 0
        total_ground_truth_chunks = 0
        
        # Create a CSV file for this evaluation run
        csv_file = os.path.join(
            self.log_path, 
            f"ground_truth_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        with open(csv_file, 'w', newline='', encoding="utf-8") as csvfile:
            fieldnames = [
                'question', 'total_expected', 'chunks_found', 'accuracy', 
                'precision', 'recall', 'f1_score', 'response_time_ms', 'top_score', 'num_docs',
                'chunk_size', 'top_k'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Create a log file for this evaluation run
            ground_truth_log_file = os.path.join(
                self.log_path, 
                f"ground_truth_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            with open(ground_truth_log_file, 'w', encoding="utf-8") as log_file:
                log_file.write("=== GROUND TRUTH EVALUATION RESULTS ===\n")
                log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Chunk Size: {self.chatbot.chunk_size}\n")
                log_file.write(f"Top K: {self.chatbot.top_k}\n")
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
                        
                        # Calculate metrics
                        total_expected = len(expected_chunks)
                        accuracy = (chunks_found / total_expected * 100) if total_expected > 0 else 0
                        
                        # Calculate precision, recall, and F1 score
                        precision = chunks_found / eval_result["evaluation"]["num_docs_retrieved"] if eval_result["evaluation"]["num_docs_retrieved"] > 0 else 0
                        recall = chunks_found / total_expected if total_expected > 0 else 0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        # Update totals for overall accuracy
                        total_ground_truth_chunks += total_expected
                        total_chunks_found += chunks_found
                        
                        # Save results to CSV
                        writer.writerow({
                            'question': question,
                            'total_expected': total_expected,
                            'chunks_found': chunks_found,
                            'accuracy': round(accuracy, 2),
                            'precision': round(precision, 2),
                            'recall': round(recall, 2),
                            'f1_score': round(f1_score, 2),
                            'response_time_ms': eval_result["evaluation"]["response_time_ms"],
                            'top_score': eval_result["evaluation"]["top_doc_score"] if eval_result["evaluation"]["top_doc_score"] else 0,
                            'num_docs': eval_result["evaluation"]["num_docs_retrieved"],
                            'chunk_size': self.chatbot.chunk_size,
                            'top_k': self.chatbot.top_k
                        })
                        
                        # Build result dictionary
                        result_dict = {
                            "question": question,
                            "expected_chunks": expected_chunks,
                            "retrieved_chunks": retrieved_chunks,
                            "answer": eval_result["result"]["answer"],
                            "chunks_found": chunks_found,
                            "total_expected": total_expected,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall, 
                            "f1_score": f1_score,
                            "found_chunks": found_chunks,
                            "missing_chunks": missing_chunks,
                            "num_docs": eval_result["evaluation"]["num_docs_retrieved"],
                            "top_score": eval_result["evaluation"]["top_doc_score"],
                            "response_time_ms": eval_result["evaluation"]["response_time_ms"],
                            "chunk_size": self.chatbot.chunk_size,
                            "top_k": self.chatbot.top_k
                        }
                        
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
                        log_file.write(f"Precision: {precision:.2f}\n")
                        log_file.write(f"Recall: {recall:.2f}\n")
                        log_file.write(f"F1 Score: {f1_score:.2f}\n")
                        log_file.write(f"Missing Chunks: {missing_chunks}\n")
                        log_file.write("-" * 50 + "\n\n")
                    except Exception as e:
                        log_file.write(f"ERROR processing test case {i+1}: {str(e)}\n")
                        log_file.write("-" * 50 + "\n\n")
                        continue
                
                # Calculate overall metrics
                overall_accuracy = (total_chunks_found / total_ground_truth_chunks * 100) if total_ground_truth_chunks > 0 else 0
                overall_precision = sum(r["precision"] for r in results) / len(results) if results else 0
                overall_recall = sum(r["recall"] for r in results) / len(results) if results else 0
                overall_f1 = sum(r["f1_score"] for r in results) / len(results) if results else 0
                
                # Log summary
                log_file.write("=== SUMMARY STATISTICS ===\n")
                log_file.write(f"Total Test Cases: {len(test_cases)}\n")
                log_file.write(f"Total Expected Chunks: {total_ground_truth_chunks}\n")
                log_file.write(f"Total Chunks Found: {total_chunks_found}\n")
                log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
                log_file.write(f"Overall Precision: {overall_precision:.2f}\n")
                log_file.write(f"Overall Recall: {overall_recall:.2f}\n")
                log_file.write(f"Overall F1 Score: {overall_f1:.2f}\n")
                log_file.write(f"Chunk Size: {self.chatbot.chunk_size}\n")
                log_file.write(f"Top K: {self.chatbot.top_k}\n")
                
                # Write summary row to CSV
                writer.writerow({
                    'question': 'OVERALL',
                    'total_expected': total_ground_truth_chunks,
                    'chunks_found': total_chunks_found,
                    'accuracy': round(overall_accuracy, 2),
                    'precision': round(overall_precision, 2),
                    'recall': round(overall_recall, 2),
                    'f1_score': round(overall_f1, 2),
                    'response_time_ms': sum(r["response_time_ms"] for r in results) / len(results) if results else 0,
                    'top_score': 'N/A',
                    'num_docs': 'N/A',
                    'chunk_size': self.chatbot.chunk_size,
                    'top_k': self.chatbot.top_k
                })
        # Create summary
        summary = {
            "total_test_cases": len(test_cases),
            "total_expected_chunks": total_ground_truth_chunks,
            "total_chunks_found": total_chunks_found,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "chunk_size": self.chatbot.chunk_size,
            "top_k": self.chatbot.top_k
        }
        
        # Prepare simplified DataFrame for display and CSV export
        display_results = []
        for result in results:
            display_result = {
                "question": result["question"],
                "answer": result["answer"],
                "expected_chunks": len(result["expected_chunks"]),
                "retrieved_chunks": len(result["retrieved_chunks"]),
                "accuracy": f"{result['accuracy']:.2f}%",
                "precision": f"{result['precision']:.2f}",
                "recall": f"{result['recall']:.2f}",
                "f1_score": f"{result['f1_score']:.2f}",
                "found_chunks": result["chunks_found"],
                "total_expected": result["total_expected"],
                "chunk_size": result["chunk_size"],
                "top_k": result["top_k"]
            }
            display_results.append(display_result)
        
        print(f"Evaluation completed. Results saved to {csv_file}")
        return pd.DataFrame(display_results), summary, results

def create_sample_yaml(output_file="test_cases.yaml"):
    """Create a sample YAML file with test cases if it doesn't exist"""
    sample_yaml = """test_cases:
  - question: "What versions of MySQL or MariaDB are compatible with Vitess?"
    expected_chunks:
      - "Please refer to our [Supported Databases](https://vitess.io/docs/overview/supported-databases/) for the most up-to-date information."
  - question: "What does it mean for Vitess to be MySQL compatible?"
    expected_chunks:
      - "Vitess supports much of MySQL, with some limitations. Depending on your MySQL setup, you will need to adjust queries that utilize any of the current unsupported cases."
      - "For SQL syntax, there is a list of example [unsupported queries](https://github.com/vitessio/vitess/blob/main/go/vt/vtgate/planbuilder/testdata/unsupported_cases.json)."
      - "There are some further [compatibility issues](https://vitess.io/docs/reference/mysql-compatibility/) beyond pure SQL syntax."
"""
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sample_yaml)
        print(f"Created sample YAML file: {output_file}")
    return output_file

def run_parameter_sweep(doc_paths, yaml_file, embedding_models=None, chunk_sizes=None, chunk_overlaps=None, top_ks=None):
    """Run evaluations with different parameter combinations"""
    if embedding_models is None:
        embedding_models = ["sentence-transformers/all-MiniLM-L6-v2"]
    if chunk_sizes is None:
        chunk_sizes = [300, 500, 700]
    if chunk_overlaps is None:
        chunk_overlaps = [0, 50, 100]
    if top_ks is None:
        top_ks = [3, 5, 7]
    
    # Initialize LLM
    llm = initialize_inference_llm()
    
    # Create results directory
    results_dir = f"param_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create aggregated results CSV
    agg_csv_path = os.path.join(results_dir, "aggregated_results.csv")
    with open(agg_csv_path, 'w', newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            'embedding_model', 'chunk_size', 'chunk_overlap', 'top_k',
            'total_test_cases', 'total_expected_chunks', 'total_chunks_found',
            'overall_accuracy', 'overall_precision', 'overall_recall', 'overall_f1',
            'avg_response_time_ms'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop through all parameter combinations
        for embedding_model in embedding_models:
            for chunk_size in chunk_sizes:
                for chunk_overlap in chunk_overlaps:
                    for top_k in top_ks:
                        print(f"\nTesting parameters: model={embedding_model}, chunk_size={chunk_size}, "
                              f"chunk_overlap={chunk_overlap}, top_k={top_k}")
                        
                        # Initialize chatbot with these parameters
                        chatbot = VitessFAQChatbot(
                            embedding_model=embedding_model,
                            llm_model=llm,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k
                        )
                        
                        # Ingest documents
                        chatbot.ingest_documents(doc_paths)
                        
                        # Create evaluator
                        evaluator = SimpleChatbotEvaluator(chatbot, 
                                                          log_path=os.path.join(results_dir, 
                                                                               f"model_{embedding_model.split('/')[-1]}_"
                                                                               f"cs{chunk_size}_"
                                                                               f"co{chunk_overlap}_"
                                                                               f"tk{top_k}"))
                        
                        # Run evaluation
                        _, summary, _ = evaluator.evaluate_with_ground_truth(yaml_file)
                        
                        # Add to aggregated results
                        writer.writerow({
                            'embedding_model': embedding_model,
                            'chunk_size': chunk_size,
                            'chunk_overlap': chunk_overlap,
                            'top_k': top_k,
                            'total_test_cases': summary.get("total_test_cases", 0),
                            'total_expected_chunks': summary.get("total_expected_chunks", 0),
                            'total_chunks_found': summary.get("total_chunks_found", 0),
                            'overall_accuracy': round(summary.get("overall_accuracy", 0), 2),
                            'overall_precision': round(summary.get("overall_precision", 0), 2),
                            'overall_recall': round(summary.get("overall_recall", 0), 2),
                            'overall_f1': round(summary.get("overall_f1", 0), 2),
                            'avg_response_time_ms': round(summary.get("avg_response_time", 0), 2)
                        })
                        
                        # Save memory by explicitly deleting objects
                        del chatbot
                        del evaluator
    
    print(f"\nParameter sweep completed. Results saved to {results_dir}")
    print(f"Aggregated results available at {agg_csv_path}")
    
    # Load and return the aggregated results
    return pd.read_csv(agg_csv_path)

def main():
    """Main function to run the evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Vitess FAQ Chatbot with varying parameters")
    parser.add_argument("--doc_paths", nargs="+", default=["./data/troubleshoot.txt", "./data/faq.txt","data/design-docs.txt","data/flags.txt","data/v22.txt"], 
                        help="Paths to documents to ingest")
    parser.add_argument("--yaml_file", default="test_cases.yaml", 
                        help="Path to YAML file with test cases")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use")
    parser.add_argument("--chunk_size", type=int, default=500, 
                        help="Chunk size for single evaluation")
    parser.add_argument("--chunk_overlap", type=int, default=75, 
                        help="Chunk overlap for single evaluation")
    parser.add_argument("--top_k", type=int, default=4, 
                        help="Top K documents for single evaluation")
    parser.add_argument("--sweep", action="store_true", 
                        help="Run parameter sweep instead of single evaluation")
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=[300, 500, 700], 
                        help="Chunk sizes for parameter sweep")
    parser.add_argument("--chunk_overlaps", nargs="+", type=int, default=[0, 50, 100], 
                        help="Chunk overlaps for parameter sweep")
    parser.add_argument("--top_ks", nargs="+", type=int, default=[3, 5, 7], 
                        help="Top K values for parameter sweep")
    
    args = parser.parse_args()
    
    # Create sample YAML file if it doesn't exist
    if not os.path.exists(args.yaml_file):
        create_sample_yaml(args.yaml_file)
    
    # Run parameter sweep if requested
    if args.sweep:
        results_df = run_parameter_sweep(
            args.doc_paths, 
            args.yaml_file,
            embedding_models=[args.embedding_model],
            chunk_sizes=args.chunk_sizes,
            chunk_overlaps=args.chunk_overlaps,
            top_ks=args.top_ks
        )
        print("Parameter sweep complete. Best configurations:")
        
        # Show top 3 configurations by accuracy
        top_configs = results_df.sort_values(by='overall_accuracy', ascending=False).head(3)
        print("\nTop configurations by accuracy:")
        print(top_configs[['chunk_size', 'chunk_overlap', 'top_k', 'overall_accuracy', 'overall_f1']])
        
        # Show top 3 configurations by F1 score
        top_f1_configs = results_df.sort_values(by='overall_f1', ascending=False).head(3)
        print("\nTop configurations by F1 score:")
        print(top_f1_configs[['chunk_size', 'chunk_overlap', 'top_k', 'overall_f1', 'overall_accuracy']])
        
    else:
        # Run single evaluation with specified parameters
        print(f"Running single evaluation with chunk_size={args.chunk_size}, "
              f"chunk_overlap={args.chunk_overlap}, top_k={args.top_k}")
        
        # Initialize LLM
        llm = initialize_inference_llm()
        
        # Initialize chatbot
        chatbot = VitessFAQChatbot(
            embedding_model=args.embedding_model,
            llm_model=llm,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k
        )
        
        # Ingest documents
        chatbot.ingest_documents(args.doc_paths)
        
        # Create evaluator
        evaluator = SimpleChatbotEvaluator(chatbot)
        
        # Run evaluation
        display_df, summary, _ = evaluator.evaluate_with_ground_truth(args.yaml_file)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total Test Cases: {summary['total_test_cases']}")
        print(f"Total Expected Chunks: {summary['total_expected_chunks']}")
        print(f"Total Chunks Found: {summary['total_chunks_found']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
        print(f"Overall Precision: {summary['overall_precision']:.2f}")
        print(f"Overall Recall: {summary['overall_recall']:.2f}")
        print(f"Overall F1 Score: {summary['overall_f1']:.2f}")

if __name__ == "__main__":
    main()
