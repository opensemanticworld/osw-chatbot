# osw-chatbot
Collection of tools to simplify the users contribution to and interaction with large knowledge graphs / linked-data-platforms (reference implementation: [OpenSemanticLab](https://github.com/OpenSemanticLab))

Features in work:
* RAG-and Graph-RAG
* panel ui component that executes client-side toolcalls
* Wrapper for an OpenAI-API providing providing schema based (see [OO-LD](https://github.com/OO-LD/schema)) structured output, file-based context and web search (osw-openai-api-wrapper)

## Demos

https://github.com/user-attachments/assets/614f33cc-2e34-434d-9293-92ee72e15eb1
> RAG and GraphRAG enhanced search interface


https://github.com/user-attachments/assets/8f3857cd-4879-4591-9fd9-b3db0e641c7c
> Panel ui component that executes client-side toolcalls: Search the right concept schema and open the auto-generated form editor


https://github.com/user-attachments/assets/8760397a-3089-4758-b480-d2cee9463234
> AI assisted form-completion based on a uploaded data sheet

## Run
```bash
git clone https://github.com/opensemanticworld/osw-chatbot 
cd osw-chatbot
cp .env.example .env
```

adapt `.env` then run

```bash
docker compose up osw-chatbot
```

or

```bash
docker compose up osw-openai-api-wrapper
```

## Development

### Chatbot App

modify and run
`src/osw_chatbot/main.py`

for integration into OpenSemanticLab see [Extension:Chatbot](https://github.com/opensemanticworld/mediawiki-extensions-Chatbot)

### Structured Output API Wrapper

modify and run
`src/osw_chatbot/structured_output/api.py`

for integration into OpenSemanticLab see [Extension:MwJson](https://github.com/opensemanticlab/mediawiki-extensions-MwJson)
