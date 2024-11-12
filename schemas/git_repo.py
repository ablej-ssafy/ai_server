from pydantic import BaseModel
from typing import Optional

class ContentRequest(BaseModel):
    content: str

class GitRepoRequest(BaseModel):
    request_id: str
    owner: str
    repo: str
    branch: str
    token: Optional[str] = None
    email: str

class GitRepoFileRequest(GitRepoRequest):
    file_path: Optional[str] = None