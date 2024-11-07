import requests
import base64

def fetch_repo_files(owner, repo, branch, token=None):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    files = response.json().get("tree", [])
    file_paths = [file["path"].strip() for file in files if file["type"] == "blob"]
    return file_paths

def fetch_file_content(owner, repo, file_path, branch, token=None):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    file_content = response.json().get("content", "")
    content_bytes = base64.b64decode(file_content + '==')
    return content_bytes.decode("utf-8")