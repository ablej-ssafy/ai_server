from fastapi import APIRouter

router = APIRouter(
    prefix="/repo",
)

@router.post("")
async def repo(
        owner: str,
        repo: str,
        branch: str,
        token: str = None
):
    form_data = {
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "token": token
    }

    return form_data