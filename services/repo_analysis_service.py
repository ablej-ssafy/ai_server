from utils.git_utils import fetch_repo_files, fetch_file_content
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from core.config import settings
import openai
import tiktoken
import torch
import re

MODEL_NAME = settings.ANALYSIS_LLM_MODEL
DEVICE = f"cuda:{settings.DEVICE_NUM}" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN).to(DEVICE)

MAX_TOKENS = model.config.max_position_embeddings

exclude_patterns = [
    "package-lock.json",
    ".git/",
    "node_modules/",
    "*.log",
    "*.tmp"
]

def is_excluded(file_path):
    for pattern in exclude_patterns:
        if re.search(pattern, file_path):
            return True
    return False

def preprocess_content(content):
    """
    content 내 불필요한 공백과 줄바꿈을 제거하여 토큰 수를 줄입니다.
    """
    # \n, \t 및 중복 공백 제거
    cleaned_content = re.sub(r'\s+', ' ', content.replace("\n", " ").replace("\t", " "))
    return cleaned_content

def split_into_chunks(content: str, max_chunk_tokens: int) -> List[str]:
    """
    텍스트를 최대 토큰 크기에 맞춰 청크로 나눕니다.
    """
    tokens = tokenizer(content, return_tensors="pt", truncation=True).input_ids[0]
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
    cleaned_content = preprocess_content(content)

    chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS)
    unique_chunks = []
    results = []

    for chunk in chunked_texts:
        if unique_chunks.count(chunk) < 3:
            unique_chunks.append(chunk)

    for chunk in unique_chunks:
        prompt = (
            f"Summarize the essential aspects of this code, focusing only on the following:\n"
            f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
            f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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
    cleaned_content = preprocess_content(content)

    chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS)
    unique_chunks = []
    results = []

    # 중복 청크를 제외하고 3번 이하의 반복만 포함
    for chunk in chunked_texts:
        if unique_chunks.count(chunk) < 2:
            unique_chunks.append(chunk)

    for chunk in unique_chunks:
        prompt = (
            f"Summarize the essential aspects of this code, focusing only on the following:\n"
            f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
            f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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

async def analyze_files(owner, repo, branch, token=None):
    files = fetch_repo_files(owner, repo, branch, token)
    analysis_results = {}

    for file_path in files:
        print(f"LOG: Analyzing file {file_path}")

        try:
            content = fetch_file_content(owner, repo, file_path, branch, token)
            if content is None:
                print(f"LOG Skipping {file_path} due to missing content.")
                continue

            # LLaMA 모델을 사용하여 요약 생성
            summary = summarize_code_with_llama(content)['final_summary']
            analysis_results[file_path] = {"summary": summary}


        except UnicodeDecodeError as e:
            print(f"LOG Failed to analyze {file_path} due to encoding error: {e}")
            analysis_results[file_path] = {"error": "Encoding error"}
        except Exception as e:
            print(f"LOG Failed to analyze {file_path}: {e}")
            analysis_results[file_path] = {"error": str(e)}

    return analysis_results


def summation_repo_codes(file_summaries):
    combined_original_summary = " ".join(summary for summary in file_summaries.values() if summary)

    prompt = (
            "The following is a combined summary of multiple code files. Extract key details relevant for a resume, "
            "focusing on purpose, main functions, technologies used, and any optimization or efficiency "
            "improvements across all files.\n\nCombined Summary:\n" + combined_original_summary +
            "\n\nResponse:"
    )

    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    summary_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary_result


def generate_openai_summary(content, directory_structure, example_summary):
    # 프롬프트 텍스트 정의
    prompt = (
        f"다음 프로젝트의 내용을 기반으로 이력서 작성에 도움이 되는 프로젝트 요약을 작성해줘.\n\n"
        f"1. 프로젝트 요약\n2. 사용 기술\n3. 핵심 기능과 서비스의 강점\n\n"
        f"디렉터리 구조:\n{directory_structure}\n\n"
        f"프로젝트 파일 요약:\n{example_summary}\n\n"
        f"참고:\n{content}\n\n"
    )

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    tokens = encoding.encode(prompt)
    token_count = len(tokens)

    print(f"LOG: token_count: {token_count}")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "응답은 한글로 작성해 주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5,
        n=1,
        stop=None
    )

    summary_text = response.choices[0]['message']['content'].strip()
    return summary_text

def project_summation(example_summary, directory_structure):
    resume_summary = generate_openai_summary(
        content="",
        directory_structure=directory_structure,
        example_summary=example_summary, indent=4, ensure_ascii=False)

    return resume_summary