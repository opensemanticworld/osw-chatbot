import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# going to do load_dotenv() here
# as OLLAMA_HOST needs to be in the environment
# before the imports below
env_path = Path(__file__).parent.parent.parent.parent / ".env"
env_loaded = load_dotenv(env_path, verbose=True)
if not env_loaded:
    print(f"No .env file found at {env_path}, using environment variables.")

from osw_chatbot.graphrag.indexer import index
from osw_chatbot.graphrag.query import local_search

from osw_chatbot.rag.langchain_rag import index as rag_index
from osw_chatbot.rag.langchain_rag import graph as rag_query

logging.basicConfig(filename="my.log", level=logging.DEBUG)

test_folder = Path(__file__).parent
output_dir = test_folder / "temp"
cache_dir = test_folder / "temp" / "cache"
test_input = test_folder / "test-data" / "lorem_ipsum.txt"


def test_rag_index():
    rag_index(input_file=test_input)


def test_rag_query():
    query = "Lorem Ipsum comes from which sections?"
    expected = "1.10.33"

    res = rag_query.invoke({"question": query})["answer"]
    print(res)
    assert expected in res


def test_index():
    artifacts = index(input_file=test_input, output_dir=output_dir, cache_dir=cache_dir)

    # check if the pandas dataframe contains a row with title="RICHARD MCCLINTOCK" with type="PERSON"
    # print(artifacts.entities)
    assert (
        artifacts.entities[
            (artifacts.entities["title"] == "RICHARD MCCLINTOCK")
            & (artifacts.entities["type"] == "PERSON")
        ].shape[0]
        > 0
    )


def test_query():
    search_chain = local_search(output_dir=output_dir, cache_dir=cache_dir)
    query = "Lorem Ipsum comes from which sections?"
    expected = "1.10.33"

    res = search_chain.invoke(query, config={"tags": ["local-search"]})
    print(res)
    assert expected in res


if __name__ == "__main__":
    test_rag_index()
    test_rag_query()
    test_index()
    test_query()
