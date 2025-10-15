import os
import requests
import base64
import json
from pprint import pprint
#from jsonref import replace_refs

from osw_chatbot.structured_output.util import modify_schema, data_url_to_text

def get_llm_response_azure_openai(promt, schema_dict = None, data_dict = None, files = None, web_search = False):

    org_promt = promt
    # Configuration
    API_KEY = os.environ['OPENAI_API_KEY']
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

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

    ENDPOINT = os.environ['OPENAI_API_ENDPOINT']

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

