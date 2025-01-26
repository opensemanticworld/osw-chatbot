from pprint import pprint
from openai import get_llm_response_azure_openai

def test_math_solution_in_steps():

    test_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                        "output": {"type": "string"}
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": False
                }
            },
            "final_answer": {"type": "string"},
                "street":{"$ref":"#/properties/steps/items/properties/explanation"},
        },
        "required": ["steps", "final_answer"],
        "additionalProperties": False
    }
    test_prompt = "solve 8x + 31 = 2"
    res = get_llm_response_azure_openai(test_promt, test_schema, files)
    pprint(res)
    
def test_address_completion():
    
    test_schema = {
        #"additionalProperties": False,"required": ["street", "city"],
        "type":"object",
        "allOf": [{ "type":"object",
        "properties":{
        "street":{"type":"string"},
        "house_number":{"type":"string"},
        "postal_code":{"type":"string"},
        "city":{"type":"string"}#,
        }},{ "type":"object","properties":{
        "country":{"type":"string"}
        }
        }]
    }
    test_promt = "Brandenburger Tor"
    res = get_llm_response_azure_openai(test_promt, test_schema, files)
    pprint(res)
    
def test_address_completion():

    test_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                "text",
                "lang"
                ],
                "properties": {
                "text": {
                    "title": "Text",
                    "type": "string",
                    "minLength": 1
                },
                "lang": {
                    "title": "Lang code",
                    "type": "string",
                    "enum": [
                    "en",
                    "de"
                    ],
                    "default": "en"
                }
                }
            }
            }

    _test_promt = "Provide a description for Elefant"

    test_promt = "What is a panguas?"
    files = None
    test_schema = None
    files = [{"name": "test.txt", "data_url": "data:/plain;base64,QSBwYW5ndWFzIGlzIGEgc21hbGwgYW5pbWFs"}]
    
    res = get_llm_response_azure_openai(test_promt, test_schema, files)
    pprint(res)

    
def test_address_completion():
    
    # Example schema
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zipcode": {"type": "string"}
                }
            },
            "contacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "value": {"type": "string"}
                    }
                }
            }
        },
        "required": ["id"]
    }


    #!pip install textract
    import io
    import base64
    from textract import process
    import tempfile
    import urllib.request
    from pprint import pprint

    def data_url_to_text(file_name, data_url):
        response = urllib.request.urlopen(data_url)
        #decoded_bytes = base64.b64decode(data_url.split("base64,")[-1])
        #decoded_file = io.BytesIO(decoded_bytes)
        result = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_name.split(".")[-1]) as temp_file:
            with temp_file.file as file:
                file.write(response.file.read())
            #temp_file.write(decoded_file.read())
            temp_file_path = temp_file.name
            print(temp_file_path)
            # Step 5: Use the temporary file with textract
            # Step 4: Use the file-like object with textract
            # Assuming the content is plain text, specify the method accordingly
            extracted_text = process(temp_file_path)
        
            result = extracted_text.decode('utf-8')
        return result

    print(data_url_to_text("test.txt", "data:/plain;base64,QSBwYW5ndWFzIGlzIGEgc21hbGwgYW5pbWFs"))

    # Modify the schema
    modify_schema(schema)

    # Print the modified schema
    #print(json.dumps(schema, indent=2))
    