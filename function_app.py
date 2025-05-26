import logging
import os
import time
import azure.functions as func
import redis
import numpy as np
from redis.commands.search.query import Query
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Configure logging settings
logging.basicConfig(level=logging.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

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
        # Measure time taken to embed the query
        start_time = time.time()
        query_embedding = embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        embedding_time = time.time() - start_time
        logging.info(f"Embedding time: {embedding_time:.4f} seconds")

        # Prepare Redis query for KNN search
        base_query = (Query(f"*=>[KNN {top_n} @embedding $vec AS score]")
                      .sort_by("score")
                      .return_fields("content", "score")
                      .paging(0, top_n)
                      .dialect(2))
        query_params = {"vec": query_embedding_np.tobytes()}

        logging.info("--->Retrieving context from Redis")
        # Execute Redis query to retrieve context
        logging.info(f"Executing Redis query: {base_query.query_string()}")
        search_result = redis_client.ft("document_index").search(base_query, query_params)

        redis_time = time.time() - (start_time + embedding_time)
        logging.info(f"Redis access and context retrieval time: {redis_time:.4f} seconds")

        return "\n".join([doc.content for doc in search_result.docs])

    except Exception as e:
        logging.error(f"Error retrieving context from Redis: {e}")
        raise

def get_openai_response(messages, model, api_key, url_llm, request_data_params, start_time):
    """
    Generate a response using the OpenAI API with the provided parameters.

    :param messages: The conversation history to be sent to the OpenAI API.
    :param model: The name of the OpenAI model to use for generating the response.
    :param api_key: The API key for authenticating with OpenAI.
    :param url_llm: The custom base URL for the OpenAI LLM, if specified.
    :param request_data_params: Additional parameters for controlling the API's behavior.
    :return: The generated response content from OpenAI.
    """
    
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

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.get('content', '')
                if delta:
                    yield f"data: {delta}\n\n"
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            yield f"event: error\ndata: {e}\n\n"
        finally:
            total = time.time() - start_time
            logging.info(f"Total execution time: {total:.4f} seconds")
            yield "event: end\ndata: [DONE]\n\n"

    return func.HttpResponse(
        body=event_stream(),
        status_code=200,
        mimetype="text/event-stream; charset=utf-8"
    )

@app.route(route="RAG", methods=["POST"])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint to process requests for generating responses using RAG (Retrieve and Generate).

    :param req: The HTTP request containing the necessary parameters for processing.
    :return: HTTP response with the generated content or an error message.
    """
    logging.info('Processing HTTP POST request')

    # Measure total execution time of the function
    start_time = time.time()

    # Parse request body
    try:
        # Parse JSON body from the request
        req_body = req.get_json()
        logging.debug(f"Request body: {req_body}")
    except ValueError as e:
        logging.error(f"Error parsing request JSON: {e}")
        return func.HttpResponse(f"Invalid JSON: {e}", status_code=400)

    # Extract required parameters from the request body
    redis_host = req_body.get('redis_host')
    redis_port = req_body.get('redis_port')
    redis_password = req_body.get('redis_password')
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
        return func.HttpResponse("Missing required parameters.", status_code=400)

    # Initialize embeddings and Redis
    os.environ["OPENAI_API_KEY"] = openai_embedding_key
    embeddings = OpenAIEmbeddings(model=openai_embedding_model, openai_api_key=openai_embedding_key)
    logging.info("Connected to OpenAI Embedding Model")
    try:
        r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)
        logging.info("Connected to Redis")
    except Exception as e:
        logging.error(f"Error connecting to Redis: {e}")
        return func.HttpResponse(f"Redis connection error: {e}", status_code=500)

    # Retrieve context
    try:
        context = retrieve_context_from_redis(query, r, embeddings, top_n)
    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        return func.HttpResponse(f"Context retrieval error: {e}", status_code=500)
    finally:
        r.close()
        logging.info("Redis connection closed")

    # Build messages
    user_content = f"Siga a regra a seguir: {rule}\n\nContexto: {context}\n\nQuestion: {query}"
    if messages and messages[-1].get('role') == 'user':
        messages[-1]['content'] += "\n\n" + user_content
    else:
        messages.append({"role": "user", "content": user_content})

    # Return streaming response
    try:
        return get_openai_response(
            messages,
            openai_llm_model,
            openai_llm_key,
            url_llm,
            request_data_params,
            start_time
        )
    except Exception as e:
        logging.error(f"Error initializing streaming response: {e}")
        return func.HttpResponse(f"Streaming initialization error: {e}", status_code=500)