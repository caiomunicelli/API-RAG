import logging
import os
import time
import redis
import numpy as np
import re
import nltk
import azure.functions as func  # Mantido para tipagem
from redis.commands.search.query import Query
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from threading import Thread
from azurefunctions.extensions.http.fastapi import StreamingResponse, Request
# Adiciona tiktoken para cálculo de tokens
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)

# NLTK download on cold start
def download_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/wordnet': 'wordnet'
    }
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package)

download_nltk_resources()

# ------------------- Pré-processamento da query -------------------
def preprocess_query(text: str) -> str:
    logging.info(f"Iniciando processamento da query: {text}")
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("portuguese") + stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    processed_text = " ".join(tokens)
    logging.info(f"Query processada: {processed_text}")
    return processed_text

# ------------------- Redis -------------------
def retrieve_context_from_redis(query, redis_client, embeddings, top_n=5):
    """
    Retrieve relevant context from Redis using the provided query and embeddings.

    :param query: The search query to be embedded and used in the Redis search.
    :param redis_client: The Redis client used for accessing the Redis database.
    :param embeddings: The OpenAIEmbeddings object used for creating the query embedding.
    :param top_n: The number of top results to retrieve from Redis.
    :return: The concatenated string of relevant context retrieved from Redis.
    """
    try:
        logging.info("--->Embedding query")
        start_time = time.time()
        cleaned_query = preprocess_query(query)
        query_embedding = embeddings.embed_query(cleaned_query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        embedding_time = time.time() - start_time
        logging.info(f"Embedding time: {embedding_time:.4f} seconds")

        base_query = (Query(f"*=>[KNN {top_n} @embedding $vec AS score]")
                      .sort_by("score")
                      .return_fields("content", "score")
                      .paging(0, top_n)
                      .dialect(2))
        query_params = {"vec": query_embedding_np.tobytes()}

        logging.info("--->Retrieving context from Redis")
        search_result = redis_client.ft("document_index").search(base_query, query_params)
        context = "\n".join([doc.content for doc in search_result.docs])

        redis_time = time.time() - (start_time + embedding_time)
        logging.info(f"Redis access and context retrieval time: {redis_time:.4f} seconds")

        return context, query_embedding

    except Exception as e:
        logging.error(f"Error retrieving context from Redis: {e}")
        raise

# ------------------- Cache semântico -------------------
def retrieve_cache_from_semantic_api(embedding, semantic_cache_endpoint):
    try:
        response = requests.post(
            semantic_cache_endpoint,
            headers={"Content-Type": "application/json"},
            json={"embedding": embedding},
            timeout=10
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Erro ao recuperar cache semântico: {e}")
        return None

def write_cache_to_semantic_api(query_embedding, store_cache_endpoint, response):
    try:
        response = requests.post(
            store_cache_endpoint,
            headers={"Content-Type": "application/json"},
            json={"content": response, "embedding": query_embedding},
            timeout=10
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Erro ao gravar cache semântico: {e}")
        return None

def stream_cache_result(cache_result: str, chunk_size: int = 200):
    sentences = re.split(r'(?<=[.!?])\s+', cache_result)  # quebra por pontuação
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) + 1 <= chunk_size:
            buffer += (" " if buffer else "") + sentence
        else:
            yield f"data: {buffer}\n\n"
            buffer = sentence
    if buffer:
        yield f"data: {buffer}\n\n"
    yield "event: end\ndata: [DONE]\n\n"

        
# ------------------- OpenAI Streaming -------------------
def get_openai_response(messages, model, api_key, url_llm, request_data_params, start_time, query_embedding=None, store_cache_endpoint=None, tokens_entrada=None):
    logging.info("Initializing OpenAI client for streaming")
    params = {
        "temperature": request_data_params.get('temperature', 0.7),
        "max_tokens": request_data_params.get('max_tokens', 512),
        "top_p": request_data_params.get('top_p', 0.95),
        "frequency_penalty": request_data_params.get('frequency_penalty', 0),
        "presence_penalty": request_data_params.get('presence_penalty', 0),
        "stream": True
    }
    client = OpenAI(api_key=api_key, base_url=url_llm) if url_llm else OpenAI(api_key=api_key)
    logging.info(f"OpenAI client ready with model: {model}")

    def event_generator():
        full_response = ""
        try:
            for chunk in client.chat.completions.create(model=model, messages=messages, **params):
                delta = chunk.choices[0].delta
                text = delta.content or ""
                if text:
                    full_response += text
                    yield f"data: {text}\n\n"
        except Exception as ex:
            logging.error(f"Streaming error: {ex}")
            yield f"event: error\ndata: {ex}\n\n"
        finally:
            # grava no cache semântico em background thread
            if query_embedding and store_cache_endpoint and full_response:
                Thread(target=write_cache_to_semantic_api, args=(query_embedding, store_cache_endpoint, full_response)).start()
            total = time.time() - start_time
            logging.info(f"Total time: {total:.4f}s")
            yield "event: end\ndata: [DONE]\n\n"

    # Adiciona o header com tokens de entrada
    headers = {"tokens-entrada": str(tokens_entrada) if tokens_entrada is not None else "0"}
    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8", headers=headers)

# ------------------- Azure Function -------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="RAG", methods=["POST"])
async def main(req: Request) -> StreamingResponse:
    logging.info('Processing HTTP POST request')
    start_time = time.time()

    try:
        req_body = await req.json()
    except Exception as e:
        logging.error(f"Error parsing request JSON: {e}")
        return StreamingResponse((f"event: error\ndata: Invalid JSON: {e}\n\n" for _ in []), status_code=400, media_type="text/event-stream")

    redis_host = req_body.get('redis_host')
    redis_port = req_body.get('redis_port')
    redis_password = req_body.get('redis_password')
    semantic_cache_endpoint = req_body.get('semantic_cache_endpoint')
    store_cache_endpoint = req_body.get('store_cache_endpoint')
    openai_embedding_key = req_body.get('openai_embedding_key')
    openai_embedding_model = req_body.get('openai_embedding_model')
    openai_llm_key = req_body.get('openai_llm_key')
    openai_llm_model = req_body.get('openai_llm_model')
    url_llm = req_body.get('url_llm', '')
    query = req_body.get('query')
    rule = req_body.get('rule')
    request_data_params = req_body.get('request_data')
    top_n = req_body.get('top_n', 5)
    messages = req_body.get('messages', [])

    required = [
        redis_host, redis_port, redis_password,
        openai_embedding_key, openai_embedding_model,
        openai_llm_key, openai_llm_model,
        query, rule, request_data_params
    ]
    if not all(required):
        logging.warning("Missing one or more required parameters.")
        return StreamingResponse((f"event: error\ndata: Missing required parameters\n\n" for _ in []), status_code=400, media_type="text/event-stream")

    os.environ["OPENAI_API_KEY"] = openai_embedding_key
    embeddings = OpenAIEmbeddings(model=openai_embedding_model, openai_api_key=openai_embedding_key)
    logging.info("Connected to OpenAI Embedding Model")

    try:
        r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        logging.info("Connected to Redis")
    except Exception as e:
        logging.error(f"Error connecting to Redis: {e}")
        return StreamingResponse((f"event: error\ndata: Redis connection error: {e}\n\n" for _ in []), status_code=500, media_type="text/event-stream")

    # --------- 1. tenta cache semântico ---------
    query_embedding = embeddings.embed_query(preprocess_query(query))
    cache_result = retrieve_cache_from_semantic_api(query_embedding, semantic_cache_endpoint)
    if cache_result:
        logging.info("Cache semântico encontrado.")
        r.close()
        return StreamingResponse(stream_cache_result(cache_result), media_type="text/event-stream")

    # --------- 2. Se não achou no cache, busca Redis ---------
    try:
        context, query_embedding = retrieve_context_from_redis(query, r, embeddings, top_n)
    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        r.close()
        return StreamingResponse((f"event: error\ndata: Context retrieval error: {e}\n\n" for _ in []), status_code=500, media_type="text/event-stream")
    finally:
        r.close()
        logging.info("Redis connection closed")

    # Monta mensagens para OpenAI
    user_content = f"Siga a regra a seguir: {rule}\n\nContexto: {context}\n\nQuestion: {query}"
    if messages and messages[-1].get('role') == 'user':
        messages[-1]['content'] += "\n\n" + user_content
    else:
        messages.append({"role": "user", "content": user_content})


    # --------- Cálculo de tokens de entrada
    try:
        enc = tiktoken.encoding_for_model(openai_llm_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens_entrada = len(enc.encode(user_content))

    # --------- 3. Streaming ---------
    return get_openai_response(messages, openai_llm_model, openai_llm_key, url_llm, request_data_params, start_time, query_embedding, store_cache_endpoint, tokens_entrada)
