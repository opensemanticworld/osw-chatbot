import json

import json
import copy
from copy import deepcopy

def is_object(item):
    return item is not None and isinstance(item, dict)

def is_array(item):
    return isinstance(item, list)

def is_string(item):
    return isinstance(item, str)

def is_integer(item):
    return isinstance(item, int)

def is_number(item):
    try:
        float(item)
        return True
    except ValueError:
        return False

def deep_copy(obj):
    import copy
    return copy.deepcopy(obj)

def deep_equal(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(deep_equal(x[key], y[key]) for key in x)
    return x == y

def unique_array(array):
    result = []
    for item in array:
        add = True
        for added_item in result:
            if deep_equal(added_item, item):
                add = False
                break
        if add:
            result.append(item)
    return result

def merge_deep(target, source):
    if not target:
        return source
    if not source:
        return target
    output = {}
    output.update(target)
    if is_object(target) and is_object(source):
        for key in source:
            if is_array(source[key]) and is_array(target.get(key)):
                if key not in target:
                    output[key] = source[key]
                else:
                    output[key] = unique_array(target[key] + source[key])
            elif is_object(source[key]):
                if key not in target:
                    output[key] = source[key]
                else:
                    output[key] = merge_deep(target[key], source[key])
            else:
                output[key] = source[key]
    return output

def merge_all_of(schema):
    """The most specific schema is on the root level"""    
    if isinstance(schema, dict):
        merged_schema = {}     
        
        for key, value in schema.items():           
            if key == "allOf":
                pass # handled later

            #if key == "$ref":
            #    pass # not implemented

            elif isinstance(value, dict):
                merged_schema[key] = merge_all_of(value)
            elif isinstance(value, list):
                merged_schema[key] = [merge_all_of(item) if isinstance(item, dict) else item for item in value]
            else: merged_schema[key] = value

            if key == "oneOf":
                merged_schema["anyOf"] = merged_schema.pop("oneOf")

        if "allOf" in schema:
            for super_schema in schema["allOf"]:
                # process it first
                super_schema = merge_all_of(super_schema)
                #print("merge", sub_schema, " with ", merged_schema)
                # then merge our schema over the super_schema
                merged_schema = merge_deep(super_schema, merged_schema)  
                
        return merged_schema
    return schema

def modify_schema(schema):
    """makes jsonschema OpenAI conform, see https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
    Interate over a JsonSchema recursively. add the key additionalProperties: false to every type object schema. for each object properties which is not required add the union type with null, e.g. type: [string, null]. finally, make all properties required.
    """
    schema = merge_all_of(schema)
    if isinstance(schema, dict) and schema.get("type") is not None:
        if "object" in schema.get("type"):
            schema["additionalProperties"] = False
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for prop, prop_schema in properties.items():
                if prop not in required:
                    if "type" in prop_schema:
                        if isinstance(prop_schema["type"], list):
                            if "null" not in prop_schema["type"]:
                                prop_schema["type"].append("null")
                        else:
                            prop_schema["type"] = [prop_schema["type"], "null"]
                    else:
                        prop_schema["type"] = ["null"]
                # Recursively modify nested objects
                modify_schema(prop_schema)
            schema["required"] = list(properties.keys())
        elif schema.get("type") == "array":
            items = schema.get("items")
            if items:
                modify_schema(items)
    elif isinstance(schema, list):
        for item in schema:
            modify_schema(item)
    return schema

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

def file_url_to_text(file_url) -> str:
    import requests
    response = requests.get(file_url)
    file_name = file_url.split("/")[-1]
    result = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_name.split(".")[-1]) as temp_file:
        with temp_file.file as file:
            file.write(response.content)
        temp_file_path = temp_file.name
        extracted_text = process(temp_file_path)
        result = extracted_text.decode('utf-8')
    return result