import os
import json
import yaml
import markdown
import subprocess
import shutil
from typing import List, Dict, Any, Tuple, Optional

# Environment variables for repository URLs (these should be set externally)
WEBSITE_REPO = os.environ.get("WEBSITE_REPO", "https://github.com/vitessio/website")
VITESS_REPO = os.environ.get("VITESS_REPO", "https://github.com/vitessio/vitess")

def clone_repo(repo_url: str, dest_dir: str) -> None:
    """Clone a Git repository if it doesn't already exist."""
    if not os.path.exists(dest_dir):
        print(f"Cloning {repo_url} to {dest_dir}...")
        subprocess.run(["git", "clone", repo_url, dest_dir], check=True)
    else:
        print(f"Repository {repo_url} already exists at {dest_dir}.")

def parse_markdown(file_path: str) -> Dict[str, Any]:
    """Parse a markdown file, extracting front matter and content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {}
        markdown_content = content
        
        # Extract YAML front matter if present
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1].strip())
                    markdown_content = parts[2].strip()
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML front matter in {file_path}: {e}")
        
        # Create a document object with metadata and content
        relative_path = os.path.relpath(file_path)
        return {
            "file_path": relative_path,
            "metadata": metadata,
            "content": markdown_content,
            "html": markdown.markdown(markdown_content)
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_markdown_files(repo_dir: str, subdir: str = None) -> List[Dict[str, Any]]:
    """Process all markdown files in a repository or subdirectory."""
    documents = []
    
    base_dir = repo_dir
    if subdir:
        base_dir = os.path.join(repo_dir, subdir)
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return documents
    
    print(f"Processing markdown files in {base_dir}...")
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                doc = parse_markdown(file_path)
                if doc:
                    documents.append(doc)
                    print(f"Processed {file_path}")
    
    return documents

def save_documents_to_json(documents: List[Dict[str, Any]], output_file: str) -> None:
    """Save the documents collection to a JSON file."""
    if not documents:
        print(f"Warning: No documents to save to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(documents)} documents to {output_file}")

def process_text_files(directory: str) -> List[Dict[str, Any]]:
    """Process text files in a directory as plain documents."""
    documents = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return documents
    
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = {
                    "file_path": file_path,
                    "metadata": {"type": "text", "filename": file},
                    "content": content
                }
                documents.append(doc)
                print(f"Processed text file {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return documents

def cleanup_repos(repo_dirs: List[str]) -> None:
    """Clean up cloned repositories with Windows-specific handling."""
    for repo_dir in repo_dirs:
        if os.path.exists(repo_dir):
            try:
                # On Windows, sometimes files are locked by other processes
                # Try multiple approaches
                try:
                    # First attempt: standard removal
                    shutil.rmtree(repo_dir)
                    print(f"Deleted {repo_dir}")
                except PermissionError:
                    # Second attempt: force with os.chmod
                    for root, dirs, files in os.walk(repo_dir, topdown=False):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                    
                    # Try again after changing permissions
                    shutil.rmtree(repo_dir)
                    print(f"Deleted {repo_dir} after changing permissions")
            except Exception as e:
                print(f"Warning: Could not fully delete {repo_dir}: {e}")
                print("You may need to manually delete this directory later.")
                
def main():
    # Create output directory
    output_dir = "processed_docs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define repository directories
    website_dir = "website_repo"
    vitess_dir = "vitess_repo"
    
    try:
        # Clone repositories
        clone_repo(WEBSITE_REPO, website_dir)
        clone_repo(VITESS_REPO, vitess_dir)
        
        # Process Vitess website documentation
        website_docs = []
        docs_paths = [
            "content/en/docs/22.0",
            "content/en/docs/faq",
            "content/en/docs/troubleshoot",
            "content/en/docs/design-docs"
        ]
        
        for path in docs_paths:
            website_docs.extend(process_markdown_files(website_dir, path))
        
        # Save website docs
        save_documents_to_json(website_docs, os.path.join(output_dir, "website_docs.json"))
        
        # Process flags from vitess repo
        flags_dir = os.path.join(vitess_dir, "go", "flags", "endtoend")
        flag_docs = process_text_files(flags_dir)
        save_documents_to_json(flag_docs, os.path.join(output_dir, "vitess_flags.json"))
        
        # Example of on-demand chunking
        print("\nExample of on-demand chunking:")
        all_chunks = []
        
        # Find a document with actual content for chunking demo
        sample_doc = None
        for doc in website_docs:
            if doc.get("content", "").strip():
                sample_doc = doc
                break
                
    finally:
        # Clean up repositories (comment out to keep repos for debugging)
        cleanup_repos([website_dir, vitess_dir])

if __name__ == "__main__":
    main()