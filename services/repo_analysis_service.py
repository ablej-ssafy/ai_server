from utils.git_utils import fetch_repo_files, fetch_file_content
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
from core.config import settings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import tiktoken
import torch
import re
import asyncio
import json
from fnmatch import fnmatch

openai_llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=1000, temperature=0.5, api_key=settings.OPENAI_API_KEY)

exclude_patterns = [
    "package-lock.json", ".classpath", ".gitignore", ".project", ".settings/*",
    ".git/", "*.png", "*.jpg", ".idea/*", "settings.gradle", "*.iml", "*.pptx",
    "node_modules/", "*.jar", "*.ico", "*.glb", "*.svg", "*.gif",
    "*.log", ".eslintrc.cjs", "jsconfig.json", ".eslintignore",
    "*.tmp",
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
    # 프롬프트 텍스트 정의
    prompt = (
        f"다음 프로젝트의 내용을 기반으로 이력서 작성에 도움이 되는 프로젝트 요약을 작성해줘. "
        f"요약, 기술 스택, 강점 등 각각의 항목에 대해 세부적으로 작성하고, 필요시 구체적인 예시를 들어줘. 응답은 제공된 템플릿에 맞춰서 HTML로 작성해줘."
        f"<h2>프로젝트 요약</h2>\n"
        f"<p>이 섹션에는 프로젝트의 목적과 주요 기능을 간략하게 설명해줘. 이 프로젝트가 해결하려고 하는 문제와 이를 위해 어떤 방법을 사용했는지 기술해.</p>\n"
        f"<h2>사용 기술</h2>\n"
        f"<p>여기에는 프로젝트에서 사용된 주요 기술, 라이브러리, 툴들을 나열해줘. 각각의 기술이 어떤 목적으로 사용되었는지 한두 문장으로 설명해.</p>\n"
        f"<h2>핵심 기능과 서비스의 강점</h2>\n"
        f"<p>이 섹션에는 프로젝트의 핵심 기능과 그 기능의 강점을 설명해줘. 이 기능들이 사용자에게 어떤 이점을 제공하는지 구체적으로 작성해줘.</p>"
        f"\n\n"
        f"<h3>디렉터리 구조:</h3>\n<pre>{directory_structure}</pre>\n\n"
        f"<h3>프로젝트 파일 요약:</h3>\n<pre>{example_summary}</pre>\n\n"
    )

    try:
        response = openai_llm.invoke(prompt)
        summary_text = response.content.strip()

        print(f"LOG: success generating")

        return summary_text
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