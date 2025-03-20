from pprint import pprint

from osw_chatbot.structured_output.openai import get_llm_response_azure_openai


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
                        "output": {"type": "string"},
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": False,
                },
            },
            "final_answer": {"type": "string"},
        },
        "required": ["steps", "final_answer"],
        "additionalProperties": False,
    }
    test_prompt = "solve 8x + 31 = 2. Provide the final answer as 'x = ...'. Use '/' for fractions."
    res = get_llm_response_azure_openai(test_prompt, test_schema)
    pprint(res)
    assert res["result"]["final_answer"] == "x = -29/8"


def test_address_completion():
    test_schema = {
        # "additionalProperties": False,"required": ["street", "city"],
        "type": "object",
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "house_number": {"type": "string"},
                    "postal_code": {"type": "string"},
                    "city": {"type": "string"},  # ,
                },
            },
            {"type": "object", "properties": {"country": {"type": "string"}}},
        ],
    }
    test_prompt = "Brandenburger Tor"
    res = get_llm_response_azure_openai(test_prompt, test_schema)
    pprint(res)
    assert res["result"]["street"] == "Pariser Platz"
    assert res["result"]["city"] == "Berlin"


def test_data_url():
    test_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["text", "lang"],
            "properties": {
                "text": {"title": "Text", "type": "string", "minLength": 1},
                "lang": {
                    "title": "Lang code",
                    "type": "string",
                    "enum": ["en", "de"],
                    "default": "en",
                },
            },
        },
    }

    _test_prompt = "Provide a description for Elefant"

    test_prompt = "What is a panguas?"
    files = None
    test_schema = None
    files = [
        {
            "name": "test.txt",
            "data_url": "data:/plain;base64,QSBwYW5ndWFzIGlzIGEgc21hbGwgYW5pbWFs",
        }
    ]

    res = get_llm_response_azure_openai(
        test_prompt, schema_dict=test_schema, files=files
    )
    pprint(res)
    assert "small animal" in res["result"]


if __name__ == "__main__":
    test_math_solution_in_steps()
    test_address_completion()
    test_data_url()
