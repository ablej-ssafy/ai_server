from utils.git_utils import fetch_repo_files, fetch_file_content
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from core.config import settings
import torch


def split_into_chunks(text: str, max_chunk_tokens: int) -> List[str]:
    MODEL_NAME = settings.ANALYSIS_LLM_MODEL
    DEVICE = f"cuda:{settings.DEVICE_NUM}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN).to(DEVICE)

    MAX_TOKENS = model.config.max_position_embeddings
    """
    텍스트를 최대 토큰 크기에 맞춰 청크로 나눕니다.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True).input_ids[0]
    chunked_texts = []
    start = 0

    while start < len(tokens):
        chunk = tokens[start:start + max_chunk_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunked_texts.append(chunk_text)
        start += max_chunk_tokens

    return chunked_texts

def text_model_response(content: str) -> Dict[str, Any]:
    """
    파일 텍스트를 나누어 모델에 입력하고 결과를 반환합니다.
    """
    chunked_texts = split_into_chunks(content, MAX_TOKENS)
    unique_chunks = []
    results = []

    # 중복 청크를 제외하고 3번 이하의 반복만 포함
    for chunk in chunked_texts:
        if unique_chunks.count(chunk) < 3:
            unique_chunks.append(chunk)

    for chunk in unique_chunks:
        prompt = f"Summarize the purpose and key features of the following text for resume preparation:\n\n{chunk}"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        except Exception as e:
            print(f"LOG: Error encountered - {str(e)}")
            results.append("Error in generating response")

    return {
        "total_tokens": sum(len(tokenizer(chunk).input_ids) for chunk in unique_chunks),
        "chunk_size": MAX_TOKENS,
        "chunk_responses": results,
        "final_summary": " ".join(results)
    }

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

    response = prompt
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