from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, List
#from loguru import logger
from fastapi.middleware.cors import CORSMiddleware

import json
from osw_chatbot.structured_output.llm import get_llm_response_azure_openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class File(BaseModel):
    name: str
    data_url: str

class Tool(BaseModel):
    name: str
    data_url: str



class UserRequestIn(BaseModel):
    promt: Optional[str] = ""
    jsonschema: Optional[Union[str, dict]] = None
    jsondata: Optional[Union[str, dict]] = None
    files: Optional[List[File]] = None
    web_search: Optional[bool] = False
    
@app.post("/test")
def index(request: UserRequestIn):
    #logger.debug(request.text)
    #logger.debug(request.data)
    #print(request)
    
    if request.jsonschema is not None and isinstance(request.jsonschema, str) and request.jsonschema != "":
        request.jsonschema = json.loads(request.jsonschema)
    if request.jsondata is not None and isinstance(request.jsondata, str) and request.jsondata != "":
        request.jsondata = json.loads(request.jsondata)
    res = get_llm_response_azure_openai(request.promt, request.jsonschema, request.jsondata, request.files, request.web_search)
    print(res)
    return res

if __name__ == "__main__":
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, port=8003)