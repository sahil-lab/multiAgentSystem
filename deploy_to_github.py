import os
import subprocess
import getpass
import requests
import json

def run_command(command):
    """Run a shell command and return the output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {command}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def initialize_git():
    """Initialize a git repository if not already initialized"""
    if not os.path.exists(".git"):
        print("Initializing git repository...")
        run_command("git init")
        return True
    else:
        print("Git repository already initialized.")
        return True

def create_github_repo(token, repo_name, description, private):
    """Create a repository on GitHub"""
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": description,
        "private": private
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        print(f"Repository {repo_name} created successfully on GitHub!")
        return response.json()["html_url"], response.json()["clone_url"]
    else:
        print(f"Failed to create repository: {response.status_code}")
        print(response.json())
        return None, None

def add_files():
    """Add all files to git"""
    print("Adding files to git...")
    run_command("git add .")
    return True

def commit_files(message):
    """Commit files to git"""
    print("Committing files...")
    run_command(f'git commit -m "{message}"')
    return True

def add_remote(remote_url, remote_name="origin"):
    """Add a remote to git"""
    print(f"Adding remote {remote_name}...")
    run_command(f"git remote add {remote_name} {remote_url}")
    return True

def push_to_remote(remote_name="origin", branch="main"):
    """Push to remote"""
    print(f"Pushing to {remote_name}/{branch}...")
    run_command(f"git push -u {remote_name} {branch}")
    return True

def create_readme(project_name, description):
    """Create a README.md file if it doesn't exist"""
    if not os.path.exists("README.md"):
        print("Creating README.md...")
        with open("README.md", "w") as f:
            f.write(f"# {project_name}\n\n{description}\n\n")
            f.write("## Models\n\n")
            f.write("This project uses GGUF models with llama-cpp-python for inference on Windows.\n\n")
            f.write("## Installation\n\n")
            f.write("```bash\n")
            f.write("pip install -r requirements.txt\n")
            f.write("```\n\n")
            f.write("## Usage\n\n")
            f.write("1. Download a model:\n")
            f.write("```bash\n")
            f.write("python download_mistral_model.py\n")
            f.write("```\n\n")
            f.write("2. Run the interface:\n")
            f.write("```bash\n")
            f.write("python run_with_gradio.py\n")
            f.write("```\n\n")
            f.write("3. Open your browser to http://127.0.0.1:7860\n")
        return True
    else:
        print("README.md already exists.")
        return True

def create_gitignore():
    """Create a .gitignore file if it doesn't exist"""
    if not os.path.exists(".gitignore"):
        print("Creating .gitignore...")
        with open(".gitignore", "w") as f:
            f.write("# Virtual Environment\n")
            f.write("venv/\n")
            f.write("env/\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("\n")
            f.write("# Models directory\n")
            f.write("models/\n")
            f.write("D:/llm_models/\n")
            f.write("D:/hf_cache/\n")
            f.write("\n")
            f.write("# Logs\n")
            f.write("*.log\n")
            f.write("\n")
            f.write("# Jupyter Notebook\n")
            f.write(".ipynb_checkpoints\n")
        return True
    else:
        print(".gitignore already exists.")
        return True

def main():
    print("=" * 50)
    print("GitHub Repository Setup and Deployment")
    print("=" * 50)
    
    # Get repository information
    github_username = input("Enter your GitHub username: ")
    repo_name = input("Enter repository name: ")
    description = input("Enter repository description: ")
    private = input("Make repository private? (y/n): ").lower() == 'y'
    
    # Get token for GitHub API
    print("\nA GitHub personal access token is required to create the repository.")
    print("You can create one at: https://github.com/settings/tokens")
    print("Make sure it has 'repo' scope permissions.")
    token = getpass.getpass("Enter your GitHub personal access token: ")
    
    # Initialize steps
    steps = [
        ("Creating README.md", lambda: create_readme(repo_name, description)),
        ("Creating .gitignore", create_gitignore),
        ("Initializing git repository", initialize_git),
        ("Adding files to git", add_files),
        ("Creating initial commit", lambda: commit_files("Initial commit")),
        ("Creating GitHub repository", lambda: create_github_repo(token, repo_name, description, private))
    ]
    
    # Execute initialization steps
    repo_url = None
    clone_url = None
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        
        if step_name == "Creating GitHub repository":
            repo_url, clone_url = step_func()
            if not repo_url or not clone_url:
                print("Failed to create GitHub repository. Aborting.")
                return
        else:
            if not step_func():
                print(f"Failed at step: {step_name}. Aborting.")
                return
    
    # Add remote and push
    if add_remote(clone_url):
        if push_to_remote():
            print("\n" + "=" * 50)
            print(f"Successfully deployed to GitHub: {repo_url}")
            print("=" * 50)
            print("\nNext steps:")
            print("1. Clone your repository on other machines:")
            print(f"   git clone {clone_url}")
            print("2. Install dependencies:")
            print("   pip install -r requirements.txt")
            print("3. Download a model:")
            print("   python download_mistral_model.py")
            print("4. Run the interface:")
            print("   python run_with_gradio.py")

if __name__ == "__main__":
    main() 