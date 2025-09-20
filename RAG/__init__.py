import logging
import os
import time
import redis
import numpy as np
import re
import string
import nltk
import azure.functions as func  # Mantido para tipagem
from redis.commands.search.query import Query
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests


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

# ------------------- OpenAI -------------------
def get_openai_response(messages, model, api_key, url_llm, request_data_params):
    logging.info("--->Calling OpenAI API")
    try:
        client = OpenAI(api_key=api_key, base_url=url_llm) if len(url_llm) > 1 else OpenAI(api_key=api_key)
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request_data_params.get('temperature', 0.7),
            max_tokens=request_data_params.get('max_tokens', 512),
            top_p=request_data_params.get('top_p', 0.95),
            frequency_penalty=request_data_params.get('frequency_penalty', 0),
            presence_penalty=request_data_params.get('presence_penalty', 0),
        )
        response_time = time.time() - start_time
        logging.info(f"OpenAI response generation time: {response_time:.4f} seconds")
        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise

# ------------------- Main -------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing HTTP POST request')
    start_time = time.time()

    try:
        req_body = req.get_json()
        redis_host = req_body.get('redis_host')
        redis_port = req_body.get('redis_port')
        redis_password = req_body.get('redis_password')
        semantic_cache_endpoint = req_body.get('semantic_cache_endpoint')
        store_cache_endpoint = req_body.get('store_cache_endpoint')
        openai_embedding_key = req_body.get('openai_embedding_key')
        openai_embedding_model = req_body.get('openai_embedding_model')
        openai_llm_key = req_body.get('openai_llm_key')
        openai_llm_model = req_body.get('openai_llm_model')
        url_llm = req_body.get('url_llm')
        query = req_body.get('query')
        rule = req_body.get('rule')
        request_data_params = req_body.get('request_data')
        top_n = req_body.get('top_n', 5)
        messages = req_body.get('messages', [])

        if not all([redis_host, redis_port, redis_password, openai_embedding_key,
                    openai_embedding_model, openai_llm_key, openai_llm_model,
                    query, rule, request_data_params]):
            return func.HttpResponse("Missing one or more required parameters.", status_code=400)

        os.environ["OPENAI_API_KEY"] = openai_embedding_key
        embeddings = OpenAIEmbeddings(model=openai_embedding_model, openai_api_key=openai_embedding_key)

        try:
            r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)
        except Exception as e:
            return func.HttpResponse(f"Error connecting to Redis: {str(e)}", status_code=500)

        conections_time = time.time() - start_time
        logging.info(f"Function conections time: {conections_time:.4f} seconds")

        # --------- 1. tenta cache semântico ---------
        query_embedding = embeddings.embed_query(preprocess_query(query))
        cache_result = retrieve_cache_from_semantic_api(query_embedding, semantic_cache_endpoint)
        if cache_result:
            logging.info("Cache semântico encontrado.")
            return func.HttpResponse(str(cache_result), status_code=200)

        # --------- 2. Se não achou no cache, busca no Redis ---------
        try:
            context, query_embedding = retrieve_context_from_redis(query, r, embeddings, top_n)
        except Exception as e:
            return func.HttpResponse(f"Error retrieving context from Redis: {str(e)}", status_code=500)
        finally:
            r.close()

        message_content = f"Siga a regra a seguir: {rule}\n\nContexto: {context}\n\nQuestion: {query}"
        if messages and messages[-1]['role'] == 'user':
            messages[-1]['content'] += message_content
        else:
            messages.append({"role": "user", "content": message_content})

        try:
            response_content = get_openai_response(messages, openai_llm_model, openai_llm_key, url_llm, request_data_params)
        except Exception as e:
            return func.HttpResponse(f"Error calling OpenAI API: {str(e)}", status_code=502)

        # --------- 3. grava no cache se não tinha ---------
        try:
            write_cache_to_semantic_api(query_embedding, store_cache_endpoint, response_content)
        except Exception as e:
            logging.error(f"Erro ao gravar cache semântico: {e}")

        if response_content:
            total_execution_time = time.time() - start_time
            logging.info(f"Total execution time: {total_execution_time:.4f} seconds")
            return func.HttpResponse(response_content, status_code=200)
        else:
            return func.HttpResponse("Response incomplete. Check parameters and try again.", status_code=502)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return func.HttpResponse(f"Unexpected error: {str(e)}", status_code=500)
