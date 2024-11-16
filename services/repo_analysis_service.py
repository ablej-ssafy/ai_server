from utils.git_utils import fetch_repo_files, fetch_file_content
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
from core.config import settings
from schemas.git_repo import ProjectSummary
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import tiktoken
import torch
import re
import asyncio
import json
from fnmatch import fnmatch

openai_llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=1000, temperature=0.5, api_key=settings.OPENAI_API_KEY)

# openai_llm = ChatOpenAI(model_name="gpt-4", max_tokens=1000, temperature=0.5, api_key=settings.OPENAI_API_KEY)

exclude_patterns = [
    "package-lock.json", ".classpath", ".gitignore", ".project", ".settings/*",
    ".git/", "*.png", "*.jpg", ".idea/*", "settings.gradle", "*.iml", "*.pptx",
    "node_modules/", "*.jar", "*.ico", "*.glb", "*.svg", "*.gif",
    "*.log", ".eslintrc.cjs", "jsconfig.json", ".eslintignore",
    "*.tmp", ".DS_Store/*", ".docker/*",
    "*.sql",
    "*.pdf"
]


def is_excluded(file_path):
    return any(fnmatch(file_path.lower(), pattern) for pattern in exclude_patterns)


def preprocess_content(content):
    """
    content 불필요한 공백과 줄바꿈을 제거 토큰 수 줄임
    """
    cleaned_content = re.sub(r'\s+', ' ', content.replace("\n", " ").replace("\t", " "))
    return cleaned_content


def split_into_chunks(content: str, max_chunk_tokens: int, tokenizer) -> List[str]:
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


def split_text(text, chunk_size=3800, overlap=200):
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    # return text_splitter.split_text(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


async def summarize_chunk(chunk):
    prompt_template = (
        "The following is a part of a project summary. Summarize it focusing on "
        "key points such as purpose, main functions, and technologies used:\n\n"
        "{chunk}\n\nSummary:"
    )
    prompt_template = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt_template.format(chunk=chunk)
    # print(f"\n\nLOG: llm prompt : {formatted_prompt}\n\n")
    response = openai_llm.invoke(formatted_prompt)
    # print(f"\n\nLOG: llm reponse : {response}\n\n")
    return response.content.strip()


async def summarize_entire_text(text, chunk_size=3800, overlap=200):
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)

    # 비동기 병렬로 각 청크 요약
    summarized_chunks = await asyncio.gather(*(summarize_chunk(chunk) for chunk in chunks))

    # 요약된 청크들을 연결하여 최종 요약 반환
    final_summary = " ".join(summarized_chunks)
    return final_summary


async def get_repo_files(owner, repo, branch, token=None):
    return fetch_repo_files(owner, repo, branch, token)


async def get_file_content(owner, repo, file_path, branch, token=None):
    return fetch_file_content(owner, repo, file_path, branch, token)


async def summarize_code_with_openai(content: str) -> str:
    try:
        filtered_content = "\n".join(
            line for line in content.splitlines() if not line.strip().startswith(("import", "from")))
        cleaned_content = preprocess_content(filtered_content)

        prompt = (
            f"Summarize the essential aspects of this code, focusing only on the following:\n"
            f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
            f"Do not mention specific function names or code details.\n\nCode:\n{cleaned_content}\n\nSummary:"
        )

        response = await asyncio.to_thread(openai_llm.invoke, prompt)
        return response.content.strip()
    except Exception as e:
        print(f"LOG: Error during summarize_code_with_openai execution - {str(e)}")
        return "Error in generating response"


# 비동기적으로 파일 분석을 수행하는 함수
async def analyze_files_with_openai(owner: str, repo: str, branch: str, token: Optional[str] = None) -> Dict[str, str]:
    files = fetch_repo_files(owner, repo, branch, token)
    files = [res for res in files if not is_excluded(res)]
    print(f"LOG: filtered files : {files}")

    analysis_results = {}
    tasks = []

    for file_path in files:
        try:
            content = fetch_file_content(owner, repo, file_path, branch, token)
            if content is None:
                print(f"LOG Skipping {file_path} due to missing content.")
                continue

            # OpenAI 요약 함수를 비동기적으로 호출하도록 task 추가
            tasks.append(analyze_file_with_openai(file_path, content, analysis_results))
        except UnicodeDecodeError as e:
            print(f"LOG Failed to analyze {file_path} due to encoding error: {e}")
        except Exception as e:
            print(f"LOG Failed to analyze {file_path}: {e}")

    # 비동기 작업 실행 (모든 작업이 완료될 때까지 대기)
    await asyncio.gather(*tasks)
    return analysis_results


# 파일 하나를 비동기적으로 분석하는 함수
async def analyze_file_with_openai(file_path: str, content: str, analysis_results: Dict[str, str]):
    try:
        summary = await summarize_code_with_openai(content)
        analysis_results[file_path] = summary
        print(f"LOG: Analyzing success file {file_path}")
    except Exception as e:
        print(f"LOG Failed to analyze {file_path}: {e}")


# OpenAI API를 사용하는 analyze_files 함수
async def analyze_files(owner, repo, branch, token=None):
    return await analyze_files_with_openai(owner, repo, branch, token)


def summarize_code_with_llama(content, DEVICE, tokenizer, model, MAX_TOKENS):
    try:
        filtered_content = "\n".join(
            line for line in content.splitlines() if not line.strip().startswith(("import", "from")))
        cleaned_content = preprocess_content(filtered_content)

        chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS, tokenizer)
        unique_chunks = []
        results = []

        for chunk in chunked_texts:
            prompt = (
                f"Summarize the essential aspects of this code, focusing only on the following:\n"
                f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
                f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            try:
                outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True,
                                         pad_token_id=tokenizer.eos_token_id)
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
    except Exception as e:
        print(f"LOG: Error during summarize_code_with_llama execution - {str(e)}")
    finally:
        # GPU 메모리 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def analyze_files2(owner, repo, branch, token=None):
    files = fetch_repo_files(owner, repo, branch, token)
    files = [res for res in files if not is_excluded(res)]
    print(f"LOG: filtered files : {files}")
    analysis_results = {}

    # 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    MODEL_NAME = settings.ANALYSIS_LLM_MODEL
    DEVICE = f"cuda:{settings.DEVICE_NUM}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN,
                                              quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN).to(DEVICE)
    MAX_TOKENS = model.config.max_position_embeddings

    for file_path in files:
        try:
            content = fetch_file_content(owner, repo, file_path, branch, token)
            if content is None:
                print(f"LOG Skipping {file_path} due to missing content.")
                continue

            # LLaMA 모델을 사용하여 요약 생성 (import 구문이 제거된 내용으로)
            summary = summarize_code_with_llama(content, DEVICE, tokenizer, model, MAX_TOKENS)['final_summary']
            print(f"LOG: Analyzing success file {file_path}")
            analysis_results[file_path] = summary
        except UnicodeDecodeError as e:
            print(f"LOG Failed to analyze {file_path} due to encoding error: {e}")
        except Exception as e:
            print(f"LOG Failed to analyze {file_path}: {e}")

    return analysis_results


async def summation_repo_codes(file_summaries):
    try:
        combined_original_summary = " ".join(summary for summary in file_summaries.values() if summary)

        with open("combined_summary.txt", "w", encoding="utf-8") as f:
            f.write(combined_original_summary)
        return await summarize_entire_text(combined_original_summary)
    except Exception as e:
        print(f"LOG summation_repo_codes Failed: {e}")
        return str(e)


async def generate_openai_summary(content, directory_structure, example_summary):
    prompt = f"""
    다음은 소프트웨어 프로젝트 코드입니다. 이 코드를 분석하여 아래 형식에 맞게 JSON으로 요약하세요.

    요구사항:
    1. 프로젝트 요약: 프로젝트의 전반적인 목표와 주요 기능을 설명합니다. 이력서에 직접 쓸 수 있을 만큼 구체적이고 매력적인 설명을 포함하십시오.
    2. 사용 기술: 프로젝트에서 사용된 주요 기술과 그 설명을 작성합니다. 각 기술의 이름과 함께 그 기술이 프로젝트에서 어떻게 사용되었는지 상세히 서술하세요.
    3. 핵심 기능과 서비스의 강점: 프로젝트의 주요 기능과 그 장점에 대해 작성합니다. 각 기능의 이름과 설명을 포함하며, 그 기능이 사용자에게 제공하는 구체적인 이점과 문제 해결 방식도 명확하게 서술해 주세요.

    프로젝트의 디렉토리 구조:
    {directory_structure}

    프로젝트 코드 요약:
    {example_summary}

    아래 형식의 JSON 형태로 응답하세요. 각 항목은 이력서와 포트폴리오에 직접 사용할 수 있도록 상세하게 작성하십시오. (이 예시는 반드시 모든 필드를 포함해야 합니다)::
    {{
      "summation": "프로젝트의 목적과 기능을 설명하는 텍스트 예시",
      "techSkills": [
        {{
            "skill": "기술 이름 예시",
            "description": "기술 설명 예시"
        }}
      ],
      "keyFeatures": [
        {{
            "feature": "기능 이름 예시",
            "description": "기능 설명 예시"
        }}
      ]
    }}

    위의 요구사항에 따라 아래 구조 및 코드 요약 정보를 토대로 JSON 형식으로 한글로 작성하여 응답하십시오.
    """

    try:
        response = openai_llm.invoke(prompt)
        summary_text = response.content.strip()
        cleaned_summary_text = re.sub(r"```json\s*|\s*```", "", summary_text)

        print(f"LOG: summary_text {cleaned_summary_text}")

        project_summary = ProjectSummary.parse_raw(cleaned_summary_text)

        print(f"LOG: success generating {project_summary}")

        return project_summary
    except Exception as e:
        print(f"LOG: Error in project_summation - {str(e)}")
        return f"Error: {str(e)}"


async def project_summation(example_summary, directory_structure):
    try:
        resume_summary = await generate_openai_summary(
            content="",
            directory_structure=directory_structure,
            example_summary=example_summary)

        return resume_summary
    except Exception as e:
        print(f"LOG: Error in project_summation - {str(e)}")
        return f"Error: {str(e)}"