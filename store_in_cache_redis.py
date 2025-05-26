import logging
import os
import azure.functions as func
import redis
import numpy as np
import uuid
from langchain_openai import OpenAIEmbeddings

# Log format
logging.basicConfig(level=logging.INFO)

def store_in_redis(redis_client, embeddings, prompt, content):
    """
    Gera o embedding do prompt e armazena no Redis com o conteúdo associado.
    """
    try:
        embedding = embeddings.embed_query(prompt)
        embedding_np = np.array(embedding, dtype=np.float32)

        doc_id = f"doc:{str(uuid.uuid4())}"

        redis_client.hset(doc_id, mapping={
            "content": content,
            "embedding": embedding_np.tobytes()
        })

        logging.info(f"Armazenado documento {doc_id}")
        return True
    except Exception as e:
        logging.error(f"Erro ao armazenar no Redis: {e}")
        return False


def ensure_index(redis_client):
    """
    Cria o índice vetorial no Redis, se ainda não existir.
    """
    try:
        redis_client.ft("document_index").info()
        logging.info("Índice 'document_index' já existe")
    except Exception:
        redis_client.ft("document_index").create_index([
            redis.commands.search.field.VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE"
            }),
            redis.commands.search.field.TextField("content")
        ])
        logging.info("Índice 'document_index' criado")


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="StoreInCache", methods=["POST"])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Recebendo requisição para armazenamento no cache semântico")

    try:
        body = req.get_json()

        prompt = body.get("prompt")
        content = body.get("content")
        redis_host = body.get("redis_host")
        redis_port = body.get("redis_port")
        redis_password = body.get("redis_password")
        openai_key = body.get("openai_key")
        embedding_model = body.get("embedding_model", "text-embedding-3-small")

        if not all([prompt, content, redis_host, redis_port, redis_password, openai_key]):
            return func.HttpResponse("Parâmetros obrigatórios ausentes", status_code=400)

        os.environ["OPENAI_API_KEY"] = openai_key
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_key)

        r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, ssl=True)

        ensure_index(r)

        success = store_in_redis(r, embeddings, prompt, content)

        return func.HttpResponse("OK" if success else "Erro ao armazenar", status_code=200 if success else 500)

    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        return func.HttpResponse(f"Erro inesperado: {str(e)}", status_code=500)
