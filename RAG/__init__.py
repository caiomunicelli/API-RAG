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
import tiktoken

def num_tokens_from_string(string: str, model: str) -> int:
    """
    Returns the number of tokens in a text string for a specific model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))

# Configure logging settings
logging.basicConfig(level=logging.INFO)

# Download NLTK resources during cold start
def download_nltk_resources():
    """
    Downloads required NLTK resources during function initialization.
    This is called only once when the function app starts.
    """
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt_tab': 'punkt_tab',  # Legacy resource for Azure compatibility
        'corpora/wordnet': 'wordnet'
    }
    
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
            logging.info(f"NLTK resource {resource_path} already exists")
        except LookupError:
            try:
                logging.info(f"Downloading NLTK resource: {package}")
                nltk.download(package)
                logging.info(f"Successfully downloaded {package}")
            except Exception as e:
                logging.error(f"Failed to download {package}: {str(e)}")
                if package != 'punkt_tab':  # Only raise for critical resources
                    raise

# Download resources during cold start
download_nltk_resources()

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
        # Measure time taken to embed the query
        start_time = time.time()
        cleaned_query = preprocess_query(query)
        query_embedding = embeddings.embed_query(cleaned_query)
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

        # Concatenate and return the retrieved context
        context = "\n".join([doc.content for doc in search_result.docs])
        ##logging.info(f"Context retrieved: {context}")

        # Log time taken to access Redis and retrieve context
        redis_time = time.time() - (start_time + embedding_time)
        logging.info(f"Redis access and context retrieval time: {redis_time:.4f} seconds")

        return context

    except Exception as e:
        logging.error(f"Error retrieving context from Redis: {e}")
        raise

def preprocess_query(text: str) -> str:
    """
    Processa e limpa o texto da query para melhorar a qualidade da busca.
    
    Args:
        text (str): O texto original da query
        
    Returns:
        str: O texto processado e limpo
        
    Etapas:
        1. Normalização do texto (minúsculas, espaços)
        2. Remoção de caracteres especiais e pontuação
        3. Tokenização
        4. Remoção de stopwords (português e inglês)
    """
    logging.info(f"Iniciando processamento da query: {text}")
    
    # Normalização básica
    text = text.lower().strip()
    logging.debug(f"Texto normalizado: {text}")
    
    # Remove caracteres especiais e pontuação
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Normaliza espaços múltiplos
    logging.debug(f"Texto após remoção de caracteres especiais: {text}")
    
    # Tokenização
    tokens = word_tokenize(text)
    logging.debug(f"Tokens gerados: {tokens}")
    
    # Remove stopwords (português + inglês)
    stop_words = set(stopwords.words("portuguese") + stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  # Remove tokens de 1 caractere
    logging.debug(f"Tokens após remoção de stopwords: {tokens}")
    
    processed_text = " ".join(tokens)
    logging.info(f"Query processada: {processed_text}")
    
    return processed_text

def get_openai_response(messages, model, api_key, url_llm, request_data_params):
    """
    Generate a response using the OpenAI API with the provided parameters.

    :param messages: The conversation history to be sent to the OpenAI API.
    :param model: The name of the OpenAI model to use for generating the response.
    :param api_key: The API key for authenticating with OpenAI.
    :param url_llm: The custom base URL for the OpenAI LLM, if specified.
    :param request_data_params: Additional parameters for controlling the API's behavior.
    :return: The generated response content from OpenAI.
    """
    logging.info("--->Calling OpenAI API")
    
    try:
        # Initialize OpenAI client with custom or default URL
        client = OpenAI(api_key=api_key, base_url=url_llm) if len(url_llm)>1 else OpenAI(api_key=api_key)
        logging.info(f"OpenAI client initialized with model: {model}")

        # Measure time taken to generate a response
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

        # Extract and return the content of the response
        response_content = response.choices[0].message.content
        output_tokens = num_tokens_from_string(response_content, model)
        logging.info(f"Output tokens count: {output_tokens}")
        return response_content

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint to process requests for generating responses using RAG (Retrieve and Generate).

    :param req: The HTTP request containing the necessary parameters for processing.
    :return: HTTP response with the generated content or an error message.
    """
    logging.info('Processing HTTP POST request')

    # Measure total execution time of the function
    start_time = time.time()

    try:
        # Parse JSON body from the request
        req_body = req.get_json()
        logging.debug(f"Request body: {req_body}")

        # Extract required parameters from the request body
        redis_host = req_body.get('redis_host')
        redis_port = req_body.get('redis_port')
        redis_password = req_body.get('redis_password')
        openai_embedding_key = req_body.get('openai_embedding_key')
        openai_embedding_model = req_body.get('openai_embedding_model')
        openai_llm_key = req_body.get('openai_llm_key')
        openai_llm_model = req_body.get('openai_llm_model')
        url_llm = req_body.get('url_llm')  # Custom URL for LLM, if specified
        query = req_body.get('query')
        rule = req_body.get('rule')
        request_data_params = req_body.get('request_data')
        top_n = req_body.get('top_n', 5)
        messages = req_body.get('messages', [])
        logging.debug(f"Messages: {messages}")

        # Ensure all necessary parameters are provided
        if not all([redis_host, redis_port, redis_password, openai_embedding_key, openai_embedding_model, openai_llm_key, openai_llm_model, query, rule, request_data_params]):
            logging.warning("Missing one or more required parameters.")
            return func.HttpResponse("Missing one or more required parameters.", status_code=400)

        # Initialize OpenAI embeddings
        os.environ["OPENAI_API_KEY"] = openai_embedding_key
        embeddings = OpenAIEmbeddings(model=openai_embedding_model, openai_api_key=openai_embedding_key)
        logging.info("Connected to OpenAI Embedding Model")

        # Connect to Redis database
        try:
            r = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password
            )
            logging.info("Connected to Redis")
        except Exception as e:
            logging.error(f"Error connecting to Redis: {e}")
            return func.HttpResponse(f"Error connecting to Redis: {str(e)}", status_code=500)
        
        # Log about conections time
        conections_time = time.time() - start_time
        logging.info(f"Function conections time: {conections_time:.4f} seconds")

        # Retrieve context from Redis
        try:
            context = retrieve_context_from_redis(query, r, embeddings, top_n)
        except Exception as e:
            logging.error(f"Error retrieving context from Redis: {e}")
            return func.HttpResponse(f"Error retrieving context from Redis: {str(e)}", status_code=500)
        finally:
            r.close()
            logging.info("Redis connection closed")

        # Append retrieved context to messages
        message_content = f"Siga a regra a seguir: {rule}\n\nContexto: {context}\n\nQuestion: {query}"
        if messages and messages[-1]['role'] == 'user':
            messages[-1]['content'] += message_content
        else:
            messages.append({"role": "user", "content": message_content})
        
        # Log context and token counts
      #  logging.info(f"context: {context.replace(chr(10), ' ').replace(chr(13), '').strip()}")
        input_tokens = num_tokens_from_string(message_content, openai_llm_model)
        logging.info(f"Input tokens count: {input_tokens}")
        
        # Call OpenAI API to generate response
        try:
            response_content = get_openai_response(messages, openai_llm_model, openai_llm_key, url_llm, request_data_params)
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return func.HttpResponse(f"Error calling OpenAI API: {str(e)}", status_code=502)      

        # Check if response is complete and return it
        if response_content:
            logging.info("Response successfully generated")
            total_execution_time = time.time() - start_time
            logging.info(f"Total execution time: {total_execution_time:.4f} seconds")
            return func.HttpResponse(response_content, status_code=200)
        else:
            logging.warning("Response incomplete")
            return func.HttpResponse("Response incomplete. Check parameters and try again.", status_code=502)

    except ValueError as e:
        logging.error(f"Error parsing request: {e}")
        return func.HttpResponse(f"Error parsing request: {str(e)}", status_code=400)
    except PermissionError as e:
        logging.error(f"Permission denied: {e}")
        return func.HttpResponse(f"Permission denied: {str(e)}", status_code=403)
    except ConnectionError as e:
        logging.error(f"Error connecting to external service: {e}")
        return func.HttpResponse(f"Error connecting to external service: {str(e)}", status_code=502)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return func.HttpResponse(f"Unexpected error: {str(e)}", status_code=500)