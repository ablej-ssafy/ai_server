from utils.git_utils import fetch_repo_files, fetch_file_content
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from core.config import settings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import tiktoken
import torch
import re
import asyncio
from fnmatch import fnmatch

MODEL_NAME = settings.ANALYSIS_LLM_MODEL
DEVICE = f"cuda:{settings.DEVICE_NUM}" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=settings.HUGGINGFACEHUB_API_TOKEN).to(DEVICE)

openai_llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=1000, temperature=0.5)

MAX_TOKENS = model.config.max_position_embeddings

exclude_patterns = [
    "package-lock.json", ".classpath", ".gitignore", ".project", ".settings/*",
    ".git/", "*.png", "*.jpg", ".idea/*", "settings.gradle", "*.iml", "*.pptx",
    "node_modules/", "*.jar", "*.ico", "*.glb", "*.svg", "*.gif",
    "*.log",
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
    print(f"\n\nLOG: chunk : {chunk}\n\n")
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
    print(f"\n\nLOG: chunks : {chunks}\n\n")

    # 비동기 병렬로 각 청크 요약
    summarized_chunks = await asyncio.gather(*(summarize_chunk(chunk) for chunk in chunks))

    # 요약된 청크들을 연결하여 최종 요약 반환
    final_summary = " ".join(summarized_chunks)
    return final_summary


def text_model_response(content: str) -> Dict[str, Any]:
    """
    파일 텍스트를 나누어 모델에 입력하고 결과를 반환합니다.
    """

    cleaned_content = preprocess_content(content)

    chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS - 100)
    results = []
    print(f"LOG: chunked_texts : {chunked_texts}")

    for chunk in chunked_texts:
        prompt = (
            f"Summarize the essential aspects of this code, focusing only on the following:\n"
            f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
            f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        except Exception as e:
            print(f"LOG: Error encountered - {str(e)}")
            results.append("Error in generating response")

    torch.cuda.empty_cache()

    return {
        "total_tokens": sum(len(tokenizer(chunk).input_ids) for chunk in chunked_texts),
        "chunk_size": MAX_TOKENS,
        "chunk_responses": results,
        "final_summary": " ".join(results)
    }


async def get_repo_files(owner, repo, branch, token=None):
    return fetch_repo_files(owner, repo, branch, token)


async def get_file_content(owner, repo, file_path, branch, token=None):
    return fetch_file_content(owner, repo, file_path, branch, token)

# def summarize_code_with_llama(content):
#     cleaned_content = preprocess_content(content)

#     chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS)
#     unique_chunks = []
#     results = []

#     # 중복 청크를 제외하고 3번 이하의 반복만 포함
#     for chunk in chunked_texts:
#         if unique_chunks.count(chunk) < 2:
#             unique_chunks.append(chunk)

#     for chunk in unique_chunks:
#         prompt = (
#             f"Summarize the essential aspects of this code, focusing only on the following:\n"
#             f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
#             f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
#         )
#         inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

#         try:
#             outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#             result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             results.append(result)
#         except Exception as e:
#             print(f"LOG: Error encountered - {str(e)}")
#             results.append("Error in generating response")

#     torch.cuda.empty_cache()

#     return {
#         "total_tokens": sum(len(tokenizer(chunk).input_ids) for chunk in unique_chunks),
#         "chunk_size": MAX_TOKENS,
#         "chunk_responses": results,
#         "final_summary": " ".join(results)
#     }


# async def analyze_files(owner, repo, branch, token=None):
#     files = fetch_repo_files(owner, repo, branch, token)
#     print(f"LOG: files : {files}")
#     files = [res for res in files if not is_excluded(res)]
#     print(f"LOG: filtered files : {files}")
#     analysis_results = {}

#     for file_path in files:
#         print(f"LOG: Analyzing file {file_path}")
#         try:
#             content = fetch_file_content(owner, repo, file_path, branch, token)
#             if content is None:
#                 print(f"LOG Skipping {file_path} due to missing content.")
#                 continue

#             # import 제외
#             filtered_lines = [
#                 line for line in content.splitlines()
#                 if not line.strip().startswith(("import", "from"))
#             ]

#             # LLaMA 모델을 사용하여 요약 생성
#             summary = summarize_code_with_llama(content)['final_summary']
#             analysis_results[file_path] = summary
#         except UnicodeDecodeError as e:
#             print(f"LOG Failed to analyze {file_path} due to encoding error: {e}")
#         except Exception as e:
#             print(f"LOG Failed to analyze {file_path}: {e}")

#     print(f"\n\nLOG: analysis_results - {analysis_results}")
#     return analysis_results

def summarize_code_with_llama(content):
    # print(f"LOG: summarize_code_with_llama content : {content}")
    filtered_content = "\n".join(
        line for line in content.splitlines() if not line.strip().startswith(("import", "from")))
    cleaned_content = preprocess_content(filtered_content)

    chunked_texts = split_into_chunks(cleaned_content, MAX_TOKENS)
    unique_chunks = []
    results = []

    # # 중복 청크를 제외하고 3번 이하의 반복만 포함
    # for chunk in chunked_texts:
    #     if unique_chunks.count(chunk) < 2:
    #         unique_chunks.append(chunk)

    # for chunk in unique_chunks:
    for chunk in chunked_texts:
        prompt = (
            f"Summarize the essential aspects of this code, focusing only on the following:\n"
            f"1) Key functionalities, 2) Important patterns and structures, 3) Unique techniques or dependencies.\n"
            f"Do not mention specific function names or code details.\n\nCode:\n{chunk}\n\nSummary:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            # max_new_tokens으로 출력 토큰 제한 설정
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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
    print(f"LOG: files : {files}")
    files = [res for res in files if not is_excluded(res)]
    print(f"LOG: filtered files : {files}")
    analysis_results = {}

    for file_path in files:
        print(f"LOG: Analyzing file {file_path}")
        try:
            content = fetch_file_content(owner, repo, file_path, branch, token)
            if content is None:
                print(f"LOG Skipping {file_path} due to missing content.")
                continue

            # LLaMA 모델을 사용하여 요약 생성 (import 구문이 제거된 내용으로)
            summary = summarize_code_with_llama(content)['final_summary']
            analysis_results[file_path] = summary
        except UnicodeDecodeError as e:
            print(f"LOG Failed to analyze {file_path} due to encoding error: {e}")
        except Exception as e:
            print(f"LOG Failed to analyze {file_path}: {e}")

    # print(f"\n\nLOG: analysis_results - {analysis_results}")
    return analysis_results


async def summation_repo_codes(file_summaries):
    try:
        combined_original_summary = " ".join(summary for summary in file_summaries.values() if summary)

        with open("combined_summary.txt", "w", encoding="utf-8") as f:
            f.write(combined_original_summary)

        print(f"\n\nLOG: combined_original_summary : {combined_original_summary}\n\n")

        # try:
        #     prompt = (
        #         f"The following is a combined summary of multiple code files. Extract key details relevant for a resume, "
        #         f"focusing on purpose, main functions, technologies used, and any optimization or efficiency "
        #         f"improvements across all files.\n\n"
        #     )

        #     encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        #     tokens = encoding.encode(prompt)
        #     token_count = len(tokens)

        #     print(f"LOG: token_count: {token_count}")

        #     response = openai.ChatCompletion.create(
        #         model="gpt-4o-mini",
        #         messages=[
        #             {"role": "system", "content": prompt},
        #             {"role": "user", "content": "Combined Summary:\n" + combined_original_summary +
        #                                         "\n\nResponse:"}
        #         ],
        #         max_tokens=1000,
        #         temperature=0.5,
        #         n=1,
        #         stop=None
        #     )

        #     summary_text = response.choices[0]['message']['content'].strip()
        #     return summary_text
        # except Exception as e:
        #     print(f"LOG Failed {e}")
        return await summarize_entire_text(combined_original_summary)
    except Exception as e:
        print(f"LOG Failed: {e}")
        return str(e)


async def generate_openai_summary(content, directory_structure, example_summary):
    # 프롬프트 텍스트 정의
    prompt = (
        f"다음 프로젝트의 내용을 기반으로 이력서 작성에 도움이 되는 프로젝트 요약을 작성해줘.\n\n"
        f"1. 프로젝트 요약\n2. 사용 기술\n3. 핵심 기능과 서비스의 강점\n\n"
        f"디렉터리 구조:\n{directory_structure}\n\n"
        f"프로젝트 파일 요약:\n{example_summary}\n\n"
        f"참고:\n{content}\n\n"
    )

    print(f"LOG: Sending prompt to openai_llm")

    try:
        response = openai_llm.invoke(prompt)
        summary_text = response.content.strip()

        return summary_text
    except Exception as e:
        print(f"LOG: Error in project_summation - {str(e)}")
        return f"Error: {str(e)}"


# async def generate_openai_summary(content, directory_structure, example_summary):
#     # 프롬프트 텍스트 정의
#     prompt = (
#         f"다음 프로젝트의 내용을 기반으로 이력서 작성에 도움이 되는 프로젝트 요약을 작성해줘.\n\n"
#         f"1. 프로젝트 요약\n2. 사용 기술\n3. 핵심 기능과 서비스의 강점\n\n"
#         f"디렉터리 구조:\n{directory_structure}\n\n"
#         f"프로젝트 파일 요약:\n{example_summary}\n\n"
#         f"참고:\n{content}\n\n"
#     )

#     encoding = tiktoken.encoding_for_model("gpt-4o-mini")

#     tokens = encoding.encode(prompt)
#     token_count = len(tokens)

#     print(f"LOG: token_count: {token_count}")

#     # response = await openai.ChatCompletion.acreate(
#     #     model="gpt-4o-mini",
#     #     messages=[
#     #         {"role": "system", "content": "응답은 한글로 작성해 주세요."},
#     #         {"role": "user", "content": prompt}
#     #     ],
#     #     max_tokens=1000,
#     #     temperature=0.5,
#     # )
#     # summary_text = response['choices'][0]['message']['content'].strip()

#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "응답은 한글로 작성해 주세요."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=1000,
#         temperature=0.5,
#         n=1,
#         stop=None
#     )

#     summary_text = response.choices[0]['message']['content'].strip()
#     return summary_text


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