from pydantic import BaseModel
from typing import Optional, List

class ContentRequest(BaseModel):
    content: str

class GitRepoRequest(BaseModel):
    request_id: str
    owner: str
    repo: str
    branch: str
    token: Optional[str] = None
    memberId: str

class TechSkill(BaseModel):
    skill: str
    description: str

class KeyFeature(BaseModel):
    feature: str
    description: str

class ProjectSummary(BaseModel):
    summation: str
    techSkills: List[TechSkill]
    keyFeatures: List[KeyFeature]

class GitRepoFileRequest(GitRepoRequest):
    file_path: Optional[str] = None