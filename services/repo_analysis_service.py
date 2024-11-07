from utils.git_utils import fetch_repo_files, fetch_file_content

async def get_repo_files(owner, repo, branch, token=None):
    return fetch_repo_files(owner, repo, branch, token)

async def get_file_content(owner, repo, file_path, branch, token=None):
    return fetch_file_content(owner, repo, file_path, branch, token)