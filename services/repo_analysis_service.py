import os
from utils.git_utils import fetch_repo_files, fetch_file_content
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from core.config import settings

llm = LlamaCpp(model_name=settings.ANALYSIS_LLM_MODEL, device=f"cuda:{settings.DEVICE_NUM}")

def text_model_response(content: str) -> str:
    """
    입력 텍스트를 요약하여 반환합니다.
    """
    prompt_template = "Summarize the purpose and key features of the following text for resume preparation:\n\n{content}"
    prompt = PromptTemplate(input_variables=["content"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # 모델 응답 생성
    response = chain.run(content=content)
    return response

async def get_repo_files(owner, repo, branch, token=None):
    return fetch_repo_files(owner, repo, branch, token)

async def get_file_content(owner, repo, file_path, branch, token=None):
    return fetch_file_content(owner, repo, file_path, branch, token)

def summarize_code_with_llama(content):
    prompt = (
        "Summarize the key logic of the following code as it would be relevant for a resume, "
        "emphasizing core functionality, optimizations, and technologies used:\n\n"
        f"{content}\n\n"
        "Provide a numbered list of main points."
    )

    response = llm(prompt)
    return response

async def analyze_files(owner, repo, branch, token=None):
    files = fetch_repo_files(owner, repo, branch, token)
    analysis_results = {}

    for file_path in files:
        try:
            content = fetch_file_content(owner, repo, file_path, branch, token)
            if content is None:
                print(f"LOG Skipping {file_path} due to missing content.")
                continue

            # LLaMA 모델로 요약 생성
            summary = summarize_code_with_llama(content)
            analysis_results[file_path] = {"summary": summary}

        except Exception as e:
            print(f"LOG Failed to analyze {file_path}: {e}")
            analysis_results[file_path] = {"error": str(e)}

    return analysis_results