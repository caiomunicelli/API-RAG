# API de Recuperação e Geração (RAG) com Redis e OpenAI/Maritaca

**Versão:** 1.0.0  
**Data de Lançamento:** 21/09/2024

## Descrição Geral
Esta API foi projetada para integrar um sistema de Recuperação e Geração (RAG) utilizando Redis como banco de dados de busca e APIs de LLMs (Large Language Models) como OpenAI e Maritaca para gerar respostas baseadas no contexto recuperado. A API é ideal para aplicações como chatbots, assistentes virtuais e sistemas de suporte ao cliente, onde a combinação de dados contextuais com inteligência artificial é necessária.

O pipeline da API é dividido em duas fases principais:
1. **Recuperação de contexto**: Busca de documentos no Redis usando embeddings.
2. **Geração de resposta**: Uso de modelos de LLM (OpenAI ou Maritaca) para gerar respostas a partir do contexto recuperado.

## Funcionalidades Principais
- **Recuperação Inteligente de Contexto**:
  - Busca eficiente de documentos armazenados no Redis usando o método K-Nearest Neighbors (KNN).
  - Embeddings gerados via OpenAI para garantir alta precisão nas buscas.

- **Geração de Respostas com LLMs**:
  - Integração com modelos de LLM da **OpenAI** como `gpt-3.5-turbo` ou `gpt-4o-mini`.
  - Suporte a modelos de LLM da **Maritaca**, como o `sabia-3`, para geração de respostas dinâmicas e personalizadas.
  - Parâmetros customizáveis como temperatura, `max_tokens`, e penalidades de presença e frequência para ajuste fino das respostas geradas.

- **Conexão Segura com Redis**:
  - Autenticação segura com Redis e fechamento automático da conexão após cada requisição.

- **Configuração Personalizável**:
  - Definição de regras específicas para guiar as respostas do LLM.
  - Suporte a URLs customizadas para acessar a API da Maritaca em ambientes privados.

## Suporte a LLMs: OpenAI e Maritaca

A API permite o uso tanto dos modelos da **OpenAI** quanto da **Maritaca**, dependendo do caso de uso:

- **Para utilizar os modelos da OpenAI**:
  - Informe a **chave da OpenAI** e o **modelo** desejado, deixando a URL vazia. 
  - Exemplo de configuração:

    ```json
    {
      "openai_llm_key": "chave da llm OpenAI",
      "openai_llm_model": "gpt-4o",
      "url_llm": ""
    }
    ```

- **Para utilizar os modelos da Maritaca (como o `sabia-3`)**:
  - Utilize o **token da Maritaca**, forneça a **URL do LLM** e selecione o modelo apropriado.
  - Exemplo de configuração:

    ```json
    {
      "openai_llm_key": "chave da llm Maritaca",
      "openai_llm_model": "sabia-3",
      "url_llm": "https://chat.maritaca.ai/api"
    }
    ```

## Exemplo de Request (JSON)

```json
{
    "redis_host": "your_redis_host",
    "redis_port": "your_redis_port",
    "redis_password": "your_redis_password",
    "openai_embedding_key": "your_openai_embedding_key",
    "openai_embedding_model": "text-embedding-3-small",
    "openai_llm_key": "your_llm_key",
    "openai_llm_model": "gpt-4o",
    "url_llm": "",
    "query": "Explique a teoria da relatividade",
    "rule": "Não divague do tema principal",
    "request_data": {
        "do_sample": true,
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0
    },
    "top_n": 5,
    "messages": [
        {
            "role": "user",
            "content": "Por favor, me explique a teoria da relatividade."
        }
    ]
}
```

## Exemplo de cURL

```bash
curl -X POST "https://<sua-api-azure>.azurewebsites.net/api/RAG" \
-H "Content-Type: application/json" \
-d '{
    "redis_host": "your_redis_host",
    "redis_port": "your_redis_port",
    "redis_password": "your_redis_password",
    "openai_embedding_key": "your_openai_embedding_key",
    "openai_embedding_model": "text-embedding-3-small",
    "openai_llm_key": "your_llm_key",
    "openai_llm_model": "gpt-4o",
    "url_llm": "",
    "query": "Explique a teoria da relatividade",
    "rule": "Não divague do tema principal",
    "request_data": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0
    },
    "top_n": 5,
    "messages": [
        {
            "role": "user",
            "content": "Por favor, me explique a teoria da relatividade."
        }
    ]
}'
```

## Recomendação de Configuração Padrão para a LLM

Recomendamos a seguinte configuração padrão para obter uma boa geração de texto balanceando criatividade e controle:

```json
{
  "request_data": {
    "do_sample": true,
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0
  },
  "top_n": 5
}
```

### Explicação dos Campos:

- **do_sample**: Ativa a amostragem. Quando `true`, a amostragem será usada ao invés de escolher sempre o token mais provável.
- **max_tokens**: Limita o número máximo de tokens gerados na resposta.
- **temperature**: Controla o grau de aleatoriedade nas respostas. Valores mais baixos tornam as respostas mais determinísticas.
- **top_p**: Usa amostragem de núcleo (nucleus sampling), onde o modelo considera os tokens com probabilidade acumulada até `p`.
- **frequency_penalty**: Penaliza novos tokens com base na frequência deles no texto gerado até o momento (controla a repetição).
- **presence_penalty**: Penaliza novos tokens com base em se eles já aparecem no texto gerado (controla a redundância).

Com essa configuração, a LLM irá gerar respostas criativas com um bom nível de controle sobre o conteúdo.

---