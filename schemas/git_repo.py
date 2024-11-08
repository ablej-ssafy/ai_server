from pydantic import BaseModel
from typing import Optional

class ContentRequest(BaseModel):
    content: str

class GitRepoRequest(BaseModel):
    owner: str
    repo: str
    branch: str
    token: Optional[str] = None

class GitRepoFileRequest(GitRepoRequest):
    file_path: Optional[str] = None