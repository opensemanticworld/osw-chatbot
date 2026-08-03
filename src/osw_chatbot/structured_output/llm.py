import os
import requests
import base64
import json
from pprint import pprint
#from jsonref import replace_refs

from osw_chatbot.config import LLM_PROVIDER
from osw_chatbot.structured_output.util import merge_all_of, modify_schema, data_url_to_text


def _augment_promt(promt, files, web_search):
    """Append web search results and file contents to the prompt."""
    org_promt = promt

    if web_search:
        promt += "\nUse the following addtional information\n\n"
        from osw_chatbot.websearch.interative_websearch import invoke
        import asyncio
        res = asyncio.get_event_loop().run_until_complete(invoke("Search in the web for addition information that could help to resolve the following request:\n" + org_promt))
        promt += res["output"]

    if files is not None:
        promt += "\nUse the following addtional information\n"
        for file in files:
            if isinstance(file, dict): promt += "\n\n" + data_url_to_text(file["name"], file["data_url"])
            else: promt += "\n\n" + data_url_to_text(file.name, file.data_url)

    return promt


def _wrap_non_object_schema(schema_dict):
    """The APIs only accept an object at the root."""
    if schema_dict["type"] != "object":
        return {
            "type": "object",
            "required": ["__dummy_root__"],
            "properties": {
                "__dummy_root__": schema_dict
            }
        }
    return schema_dict


def _unwrap_dummy_root(res):
    if isinstance(res, dict) and "__dummy_root__" in res:
        return res["__dummy_root__"]
    return res


def get_llm_response(promt, schema_dict = None, data_dict = None, files = None, web_search = False):
    """Dispatch to the provider configured via LLM_PROVIDER."""
    if LLM_PROVIDER in ("claude", "anthropic"):
        return get_llm_response_claude_foundry(promt, schema_dict, data_dict, files, web_search)
    return get_llm_response_azure_openai(promt, schema_dict, data_dict, files, web_search)


def get_llm_response_claude_foundry(promt, schema_dict = None, data_dict = None, files = None, web_search = False):
    """Structured output from Claude on Microsoft Foundry.

    The Messages API has no `response_format`, so a schema is enforced by
    declaring it as the input schema of a single tool and forcing that tool.
    Note the OpenAI strict-mode rewriting in `modify_schema` (everything
    required, null unions) must NOT be applied here - Anthropic accepts plain
    JSON Schema and marking every property required would make the model
    invent values.
    """
    promt = _augment_promt(promt, files, web_search)

    endpoint = os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": os.environ["ANTHROPIC_MODEL"],
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        "messages": [{"role": "user", "content": promt}],
    }

    if schema_dict is not None:
        schema_dict = _wrap_non_object_schema(merge_all_of(schema_dict))
        payload["tools"] = [{
            "name": "default_schema",
            "description": "Return the requested data in the given structure.",
            "input_schema": schema_dict,
        }]
        payload["tool_choice"] = {"type": "tool", "name": "default_schema"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(response.text)
        if schema_dict is not None:
            print(json.dumps(schema_dict, indent=2))
        raise SystemExit(f"Failed to make the request. Error: {e}")

    body = response.json()
    result = {"ok": True}

    if schema_dict is not None:
        tool_use = next(
            (b for b in body.get("content", []) if b.get("type") == "tool_use"), None
        )
        if tool_use is None:
            result["ok"] = False
            result["error_msg"] = "".join(
                b.get("text", "") for b in body.get("content", [])
            ) or body.get("stop_reason")
            result["result"] = None
            return result
        result["result"] = _unwrap_dummy_root(tool_use.get("input"))
        return result

    result["result"] = "".join(
        b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"
    )
    return result


def get_llm_response_azure_openai(promt, schema_dict = None, data_dict = None, files = None, web_search = False):

    # Configuration
    API_KEY = os.environ['OPENAI_API_KEY']
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,                     # Azure OpenAI style
        "Authorization": f"Bearer {API_KEY}",   # OpenAI style (Foundry v1)
    }

    promt = _augment_promt(promt, files, web_search)


    # Payload for the request
    payload = {
        "messages": [
            {
                "role": "user",
                "content": promt,
                #"temperature": 0.1
            }
        ],
        #"temperature": 0.7,
        #"top_p": 0.95,
        #"max_tokens": 800,

    }

    # Foundry v1 surface: the deployment goes into the payload, not the url.
    # Falls back to the legacy OPENAI_API_ENDPOINT (which carries the
    # deployment name and an api-version) when OPENAI_BASE_URL is not set.
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        ENDPOINT = base_url.rstrip("/") + "/chat/completions"
        payload["model"] = os.environ["OPENAI_MODEL"]
    else:
        ENDPOINT = os.environ["OPENAI_API_ENDPOINT"]
    if schema_dict is not None:
        #schema_dict = replace_refs(schema_dict, proxies=False) # cannot handle circ refs
        schema_dict = modify_schema(schema_dict)
        # Define the JSON schema for structured output
        if schema_dict["type"] != "object":
            schema_dict = {
                "type": "object",
                "required": ["__dummy_root__"],
                "properties": {
                    "__dummy_root__": schema_dict
                }
            }
        response_format ={
            "type": "json_schema",
            "json_schema": {
                "name": "default_schema",
                "schema": schema_dict,
                "strict": False # True only works if all properties are required and no optional keywords are present
            }
        }
        payload["response_format"] = response_format

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        print(response.text)
        if schema_dict is not None: 
            print(json.dumps(schema_dict, indent=2))
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    print(response.json())
    #print(response.json()["choices"][0]["message"]["content"])
    res = response.json()["choices"][0]["message"]["content"]
    refusal = response.json()["choices"][0]["message"]["content"]
    response = {"ok": True}
    if schema_dict is not None:
        if res is not None:
            res = json.loads(res)
            if "__dummy_root__" in res:
                res = res["__dummy_root__"]
        else:
            response["ok"] = False
            response["error_msg"] = refusal
    response["result"] = res
    return response

