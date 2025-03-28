import os
import subprocess
import shutil
import platform
import time

folder_name = "data"
os.makedirs(folder_name, exist_ok=True)

def clone_repo(repo_url, dest_dir):
    if not os.path.exists(dest_dir):
        subprocess.run(["git", "clone", repo_url, dest_dir], check=True)
    else:
        print(f"Repository {repo_url} already cloned.")

def extract_markdown_content(repo_path, docs_path, output_filename):
    all_docs = []
    full_path = os.path.join(repo_path, docs_path)
    if os.path.exists(full_path):
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(".md"):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            all_docs.append(f.read())
                    except Exception as e:
                        print(f"Error reading {os.path.join(root, file)}: {e}")
    
    combined_text = "\n\n".join(all_docs)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"Extracted content saved to {output_filename}")

def combine_text_files(source_directory, output_filename):
    txt_files = [f for f in os.listdir(source_directory) if f.endswith(".txt")]
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for txt_file in txt_files:
            file_path = os.path.join(source_directory, txt_file)
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Combined {len(txt_files)} files into {output_filename}")

def safe_cleanup(path):
    """
    Safely delete a directory on Windows by handling common permission errors
    """
    def onerror(func, path, exc_info):
        # Handle permission denied errors
        print(f"Warning: Permission error when removing {path}. Skipping.")
    
    try:
        if os.path.exists(path):
            shutil.rmtree(path, onerror=onerror)
            print(f"Deleted {path}")
    except Exception as e:
        print(f"Warning: Could not delete {path}: {e}")
        print("You may need to manually delete this directory later.")

def cleanup_repos(repo_dirs):
    for repo_dir in repo_dirs:
        safe_cleanup(repo_dir)

# Define repositories and paths
repos = [
    ("https://github.com/vitessio/website", "website", [
        ("content/en/docs/22.0", "./data/v22.txt"),
        ("content/en/docs/faq", "./data/faq.txt"),
        ("content/en/docs/troubleshoot", "./data/troubleshoot.txt"),
        ("content/en/docs/design-docs", "./data/design-docs.txt")
    ]),
    ("https://github.com/vitessio/vitess", "vitess", [])
]

for repo_url, repo_dir, doc_paths in repos:
    clone_repo(repo_url, repo_dir)
    for docs_path, output_filename in doc_paths:
        extract_markdown_content(repo_dir, docs_path, output_filename)

go_flags_dir = os.path.join("vitess", "go", "flags", "endtoend")
if os.path.exists(go_flags_dir):
    # Save flags.txt in ./data directory
    combine_text_files(go_flags_dir, os.path.join("./data", "flags.txt"))
else:
    print(f"Directory {go_flags_dir} does not exist.")

# Cleanup downloaded repositories
cleanup_repos([repo[1] for repo in repos])
print("Script completed. Check the data directory for extracted files.")