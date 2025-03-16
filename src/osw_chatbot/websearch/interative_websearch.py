
# a function to search the web via startpage.com and return the results
import os
from typing import List, Optional
import langchain_core
from pydantic import BaseModel
import asyncio

import langchain_core.tools

import langchain_core

from osw_chatbot.llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore

from osw_chatbot.structured_output.util import file_url_to_text

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# read HEADLESS from env, default to True
headless = os.getenv('HEADLESS', 'true').lower() in ('true', '1')

class WebSearchParam(BaseModel):
    query: str
    """The search query a human would enter into the search bar."""
    max_results: int = 10
    """The maximum number of search results to return."""

class WebLink(BaseModel):
    url: str
    """The URL of the search result. This link can be followed to learn more."""
    title: str
    """The title of the search result."""
    description: str
    """A short description of the search result."""

class SearchResult(WebLink):
    pass

    
class WebSearchResult(BaseModel):
    results: List[SearchResult]

@langchain_core.tools.tool
async def web_search_tool(param: WebSearchParam) -> WebSearchResult:
    """Uses a search engine to search the web for the given query and returns the results.
    """
    return await web_search(param)
    
async def web_search(param: WebSearchParam) -> WebSearchResult:
    """Uses a search engine to search the web for the given query and returns the results.
    """
    
    # GET https://www.startpage.com/sp/search?query=<query>&cat=web&pl=opensearch&language=english
    url = f"https://www.startpage.com/sp/search?query={param.query}&cat=web&pl=opensearch&language=english"
    #url = "https://www.startpage.com/sp/search?query=MP%C2%B3ROVE+sinterpasten&cat=web&pl=opensearch&language=english"
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    # }
    # response = requests.get(url, headers=headers)
    # print(response.text)
    # # save the response to a file
    # with open("response.html", "w", encoding="utf-8") as f:
    #     f.write(response.text)
    
    # interate over all elements with css class "result"
    # get the title from the text value of subelements with css class "result-title"
    # get the url from the text value of subelements with css class "result-link"
    # get the description from the text value of subelements with css class "description"
    
    #from bs4 import BeautifulSoup
    #soup = BeautifulSoup(response.text, 'html.parser') 
    #results = []
    #for result in soup.find_all('div', class_='result'):
    #    title = result.find('a', class_='result-title').text
    #    url = result.find('a', class_='result-link').text
    #    description = result.find('p', class_='description').text
    #    results.append(SearchResult(url=url, title=title, description=description))
        
    # select text value per regex: <p class="description css-1507v2l">mAgic- und magiCu-<b>Sinterpasten</b> für Hochleistungsanwendungen. Höhere Leistungsdichten gehen mit höheren Betriebstemperaturen einher. Gleichzeitig müssen Geräte&nbsp;...</p>
    #regex_pattern = r'<p class=\"description[^>]*>(.+?)</p>'
    #results = []
    #for match in regex.finditer(regex_pattern, response.text):
    #    description = match.group(1)
    #    results.append(SearchResult(url="url", title="title", description=description))
    
    # use playwright to extract the data
    from playwright.async_api import async_playwright, Playwright
    async with async_playwright() as playwright:
        chromium = playwright.chromium # "chromium" or "firefox" or "webkit".
        browser = await chromium.launch(
            headless=headless, # otherwise bot detection will be triggered
        )
        if headless:
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
            browser = await browser.new_context(user_agent=user_agent, bypass_csp=True)
        page = await browser.new_page()
        await page.goto(url)
        try:
            await page.wait_for_load_state("networkidle")
        except:
            pass
        # # save the page to a file
        # with open("page.html", "w", encoding="utf-8") as f:
        #     f.write(await page.content())
        # # save the page text to a file
        # with open("page.txt", "w", encoding="utf-8") as f:
        #     f.write(await (await page.query_selector("body")).text_content())
        results = []
        for element in await page.query_selector_all(".result"):
            try:
                title = await (await element.query_selector(".result-title")).text_content()
                url = await (await element.query_selector(".result-link")).get_attribute("href")
                description = await (await element.query_selector(".description")).text_content()
                results.append(SearchResult(url=url, title=title, description=description))
            except:
                pass
        
    return results[:param.max_results]

class BrowseWebPageParam(BaseModel):
    url: str
    """The URL of the web page to browse."""
    chunk: Optional[int] = 0
    """The zero-based index of the current requested content chunk if the page content is to large to fit in a single context."""
    
class BrowseWebPageResult(BaseModel):
    text: str
    """The text content of the web page."""
    links: List[WebLink]
    """links found on the web page."""
    #rag_query: str = ""
    #"""If the page content is to large for single context, the RAG query to be used to generate a reduced text and link list."""
    chunk: Optional[int] = 0
    """The zero-based index of the current content junk the page content is to large. If < total_chunks -1 request the next chunk."""
    total_chunks: Optional[int] = 1
    """The total number of chunks of the page content."""

@langchain_core.tools.tool
async def browse_web_page_tool(param: BrowseWebPageParam) -> BrowseWebPageResult:
    """Loads the given web page and returns the text content and links.
    Links can be followed to learn more about specific topics.
    """
    return await browse_web_page(param)

cache = {}
if not "buffer" in cache: cache["buffer"] = ""

async def browse_web_page(param: BrowseWebPageParam) -> BrowseWebPageResult:
    """Loads the given web page and returns the text content and links.
    Links can be followed to learn more about specific topics.
    If the page is to large to fit in a single context, the text and links are split into chunks.
    Further chunks should be requested one after another. Processing should be done in between while storing the intermediate results to keep limit the content size.
    All chunks need to be processed to get the full content of the page.
    """
    
    if param.url in cache:
        result = cache[param.url]
    # todo: use content type to decide if the page is a file or a web page
    elif param.url.split(".")[-1] in ["doc", "docx", "pdf", "xls", "xlsx", "ppt", "pptx"]:
        text = file_url_to_text(param.url)
        result = BrowseWebPageResult(text=text, links=[])
        cache[param.url] = result
    else:
        from playwright.async_api import async_playwright, Playwright
        async with async_playwright() as playwright:
            chromium = playwright.chromium # or "firefox" or "webkit".
            browser = await chromium.launch(
                headless=headless,
            )
            if headless:
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0" # otherwise bot detection will be triggered
                browser = await browser.new_context(user_agent=user_agent, bypass_csp=True)
            page = await browser.new_page()
            await page.goto(param.url)
            # parse html page
            try:
                await page.wait_for_load_state("networkidle")
            except:
                pass
            # # save the page to a file
            # with open("page.html", "w", encoding="utf-8") as f:
            #     f.write(await page.content())
            # # save the page text to a file
            # with open("page.txt", "w", encoding="utf-8") as f:
            #     f.write(await (await page.query_selector("body")).text_content())
            text = await (await page.query_selector("body")).text_content()
            links = []
            for element in await page.query_selector_all("a"):
                try:
                    url = await element.get_attribute("href")
                    # if the url is relative, make it absolute
                    if not url.startswith("http"):
                        url = await page.evaluate('document.location.origin') + url
                    title = await element.text_content()
                    links.append(WebLink(url=url, title=title, description=""))
                except:
                    pass
        
            result = BrowseWebPageResult(text=text, links=links)
            cache[param.url] = result
        
    # if if total token length of text is larger than 4096, reduce the text and links  
    # split the text into chunks of 4096 tokens
    # if the text is larger than 4096 tokens, reduce the text and links
    from langchain_text_splitters import TokenTextSplitter

    # model max 128000 tokens
    text_splitter = TokenTextSplitter(chunk_size=10000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(result.text)
    text_chunk_count = len(text_chunks)  

    # split the links into chunks
    link_chunk = 0
    link_chunks = [[]]
    link_json =""
    for link in result.links:
        link: WebLink
        link_json += link.json()
        chunks = text_splitter.split_text(link_json)
        if len(chunks) > 1:
            link_chunk += 1
            link_chunks.append([link])
            link_json = link.json()
        else:
            link_chunks[link_chunk].append(link)
            
    link_chunk_count = len(link_chunks)

    chunks = text_chunk_count + link_chunk_count
    if chunks > 1:
        if param.chunk < text_chunk_count:
            result = BrowseWebPageResult(text=text_chunks[param.chunk], links=[], chunk=param.chunk, total_chunks=chunks)
        else:
            result = BrowseWebPageResult(text="", links=link_chunks[param.chunk - text_chunk_count], chunk=param.chunk, total_chunks=chunks)
        #result = BrowseWebPageResult(text="", links=link_chunks[param.chunk], chunk=param.chunk, total_chunks=link_chunk_count)

    return result


#buffer = ""

@langchain_core.tools.tool
async def store_intermediate_result_tool(text: str):
    """Stores a partial promt response in memory. The is relevant if a large list of websites is to be processed interatively.
    All stored text is appended to the final response.
    """
    return await store_intermediate_result(text)
    
async def store_intermediate_result(text: str):
    """Stores a partial promt response in memory. The is relevant if a large list of websites is to be processed interatively.
    All stored text is appended to the final response.
    """
    
    cache["buffer"] = cache["buffer"] + text

tools = [web_search_tool, browse_web_page_tool, store_intermediate_result_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            str((
                "You are a helpful assistant. Try to answer the user's question directly. If you need additional information, search the web for it. ",
                "If a result page contains helpful links you can follow them if needed but try to minimize the number of links you follow.",
                "If the user explicitly provides a link you must follow it to get the information.",
                "Provide the user with a summary of the information you found and, if applicable, links to the most relevant pages.",
            )),
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=None)
#agent_executor.invoke({"input": "what is LangChain?"})

from langchain_core.messages import AIMessage, HumanMessage
chat_history = []

async def invoke(prompt):
    res = {}
    try:
        res = await agent_executor.ainvoke({"input": prompt, "chat_history": chat_history})
        chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=res["output"]),
        ])
        res["output"] += cache["buffer"]
    except Exception as e:
        res["output"] = "An error occured. Details: " + str(e)
    return res

def test():
    import asyncio
    results = asyncio.run(invoke("EMMC 2025 Workshop"))

    print(results)
    

def test_tools():
    
    results = asyncio.run(web_search(WebSearchParam(query="filtech exhibitor-list", max_results=1)))
    for res in results:
        print(res.url)
        print(res.title)
        print(res.description)
        print("===")
        
        result = asyncio.run(browse_web_page(BrowseWebPageParam(url=res.url)))
        print(result.text)
        for link in result.links:
            print(link.url)
            print(link.title)
            print(link.description)
        print("===")
  
def test_pdf_download():
    result = asyncio.run(browse_web_page(BrowseWebPageParam(url="https://de.wikipedia.org/wiki/Wikipedia:Hauptseite")))
    print(result)
    result = asyncio.run(browse_web_page(BrowseWebPageParam(url="https://wiki.creativecommons.org/images/6/61/Creativecommons-licensing-and-marking-your-content_eng.pdf")))
    print(result)
          
if __name__ == "__main__":
    #test_tools()
    #test()
    test_pdf_download()
    