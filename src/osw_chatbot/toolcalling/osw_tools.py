"""this file shall contain useful functions to be used for the interaction
 with OpenSemanticLab
 many tools are found at https://github.com/opensemanticworld/mediawiki-extensions-Chatbot/blob/main/modules/ext.osw.ui.chatbot/chatbot.js
 """
import dotenv
dotenv.load_dotenv()
from langchain.tools import tool

from pydantic.v1 import BaseModel, Field
from osw.express import osw_download_file, OswExpress
from urllib.request import urlopen
from SPARQLWrapper import SPARQLWrapper, JSON
import os
from uuid import UUID
import re
import pandas as pd
osw = OswExpress(domain="mat-o-lab.open-semantic-lab.org")

class GetPageHtmlInput(BaseModel):
    fullpagetitle: str = Field(..., description="The title of the page to get the html from including the namespace. "
                                                "Example: Item:OSW70b4d6464c1d44a887eb86e3b39b8751")

@tool
def get_page_html(inp:GetPageHtmlInput):
    """Get the html content of a page from the main slot. This schould contain most of the information the user would see
     on this page (exceot for contend rendered with javascript)"""
    json_with_html = osw.site._site.raw_api(
        action="parse",
        page=inp.fullpagetitle,
        format="json")
    return(json_with_html)

class DownlaodOslFileInput(BaseModel):
    osw_id: str = Field(..., description="The id of the OSW element to download the file from. Can start with File: "
                                         "or OSW, for example File:OSW29b9f7873b6f4752beafc4cc57b65db2.csv")

@tool
def download_osl_file(inp: DownlaodOslFileInput):
    """ Download a file from an OSW instance and save it to a local file
    returns

    local_file_path: str the local path to the downloaded file"""
    ## clean up osw_id:

    try:
        if not inp.osw_id.startswith("File:"):

            if inp.osw_id.startswith("OSW"):
                inp.osw_id = "File:" + inp.osw_id
            else:
                raise ValueError("OSW id must start with 'File:' or 'OSW'")

        print("downloading ", inp.osw_id)
        local_file = osw_download_file(
            "https://mat-o-lab.open-semantic-lab.org/wiki/"+
            inp.osw_id,
            # , use_cached=True
            overwrite=True
        )
        local_file_path = local_file.path
        return(local_file_path)
    except Exception as e:
        return "could not download file, excpetion " + str(e)

# class GetShortestPathInput(BaseModel):
#     start_node: str = Field(..., description="The start node of the path")
#     end_node: str = Field(..., description="The end node of the path")
#
# @tool
# def get_shortest_path(inp: GetShortestPathInput):
#     """Get the shortest path between two nodes in a graph"""
#     print("shortest path")
#

## Sparql query that gets all unidirectional paths between two nodes
"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
        PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
        PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
        PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
        PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>

        SELECT DISTINCT #?node1 
						#?node2 
                        #?label1
                        #?label2 
                        #?uuid1 
                        #?uuid2 
                        ?labeltext1 
                        ?labeltext2 
                        #?p 
						?p_label
						#?x 
						?x_label	
						#?y 
						?y_label
						
        WHERE {
          
          ?node1 (<>|!<>)* ?x .
          ?x ?p ?y .
          ?y (<>|!<>)* ?node2 .
          ?node1 Property:HasUuid ?uuid1 .
          ?node2 Property:HasUuid ?uuid2 .
          ?node1 Property:HasNormalizedLabel ?label1 .
          ?node2 Property:HasNormalizedLabel ?label2 .
          ?p Property:HasName ?p_label .
          ?x Property:HasName ?x_label .
          ?y Property:HasName ?y_label .
          
          ?label1 <https://mat-o-lab.open-semantic-lab.org/id/Property-3AText> ?labeltext1.
          ?label2 <https://mat-o-lab.open-semantic-lab.org/id/Property-3AText> ?labeltext2.
          
          FILTER(?uuid1 = "a5fd64a4-e26e-4b7d-abdb-b8c0db83ddd6")  # Matthias Albert Popp
          #FILTER(?uuid1 = "b3a52473-87d0-4385-95e7-ecdda1f6b1af")  # Robin Pfeiffer
          
          FILTER(?uuid2 = "4240d9f1-cbe6-45bd-b932-0868584f7071") # Item
   #       FILTER(?uuid2 = "82ee4dd0-696b-4fc5-9afd-643ea6f7c10c") # Rene Wickmann
		  FILTER(?p != 	<http://semantic-mediawiki.org/swivt/1.0#masterPage>)
          FILTER(?p != 	<https://mat-o-lab.open-semantic-lab.org/id/Property-3AHas_query>)
          }"""

class GetFileHeaderInput(BaseModel):
    file_path: str = Field(description="The path to the file to get the header from.")
    n_lines: int = Field(default=10, description="The number of lines to read from the file.")

@tool
def get_file_header(inp: GetFileHeaderInput):
    """a function that reads the header of a file and returns it as text"""
    if inp.file_path.endswith(".txt") or inp.file_path.endswith(".csv"):
        with open(inp.file_path, 'r') as file:
            lines = [next(file) for _ in range(10)]
        return ''.join(lines)


class SparqlSearchFunctionInput(BaseModel):
    search_string: str = Field(...,
                               description="The search string to look for. All words inside the search string must be contained in the normalized label.")



def check_for_uuid(input_str):
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return re.match(pattern, input_str, re.IGNORECASE) is not None

def try_cast_str_to_uuid(input_str):
    if check_for_uuid(input_str):
        return input_str
    if "osw" in input_str.lower():
        str_uuid = input_str[-32:]  # Get the last 32 characters
        uuid = ""
        uuid += str_uuid[0:8] + "-"
        uuid += str_uuid[8:12] + "-"
        uuid += str_uuid[12:16] + "-"
        uuid += str_uuid[16:20] + "-"
        uuid += str_uuid[20:32]
        return uuid
    return None

@tool
def sparql_search_function(inp: SparqlSearchFunctionInput):
    """Search for a string in the Mat-O-Lab OSW."""

    sparql_url = os.environ.get("BLAZEGRAPH_ENDPOINT")
    ## check if an osw-id is contained in the search string, then directly use it
    search_string_uuid = try_cast_str_to_uuid(inp.search_string)
    if search_string_uuid is not None:
        ## directly search for elements with found uuid:
        sparql_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
        PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
        PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
        PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
        PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>

        SELECT DISTINCT ?node ?label ?labeltext ?osw_id ?uuid
        WHERE {
          ?node Property:HasUuid ?uuid .
          ?node Property:HasNormalizedLabel ?label .
          ?label <https://mat-o-lab.open-semantic-lab.org/id/Property-3AText> ?labeltext.
          
          ?node Property:HasOswId ?osw_id.
          FILTER(?uuid = \"""" + search_string_uuid + """\")
        }"""

    else:


        ## generate filter string:
        filter_string = ""
        for spl in inp.search_string.replace("-", "").split(" "):
            filter_string += "FILTER(CONTAINS(LCASE(STR(?labeltext)), LCASE(\"" + spl + "\")))\n"
        sparql_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
    PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
    PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
    PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
    PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>
    
    SELECT DISTINCT ?node ?label ?labeltext ?osw_id
    WHERE {
      ?node Property:HasNormalizedLabel ?label .
      ?label <https://mat-o-lab.open-semantic-lab.org/id/Property-3AText> ?labeltext
             """ + filter_string + """
             ?node Property:HasOswId ?osw_id
    }"""

    sparql = SPARQLWrapper(sparql_url)
    sparql.setHTTPAuth('BASIC')
    sparql.setCredentials(os.environ.get("BLAZEGRAPH_USER"), os.environ.get("BLAZEGRAPH_PASSWORD"))

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


class FindOutEverythingAboutInput(BaseModel):
    osw_id: str = Field(..., description="The id of the OSW element to find out everything about, for example for "
                                         "example File:OSW29b9f7873b6f4752beafc4cc57b65db2 ",
                                    regex=".*OSW[0-9a-f]{32}.*")
    depth: int = Field(1, description="The depth of the search. Default is 1.")


@tool
def find_out_everything_about(inp: FindOutEverythingAboutInput):
    """Get all properties of an OSW element. This can be used to further explore the element and find out more about it.
    returns the result of a sparql query going through all properties of the given element in a star shape in the
    given depth"""
    # print(inp)
    # print(inp.osw_id)

    sparql_url = os.environ.get("BLAZEGRAPH_ENDPOINT")

    osw_id = inp.osw_id.split(":")[-1].split('.')[0]  ## second split: get rid of file ending e.g. .csv
    my_uuid = str(UUID(osw_id.replace("OSW", "")))
    sparql_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
        PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
        PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
        PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
		PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>

        SELECT DISTINCT ?s ?p ?o ?s_label ?p_label ?o_label
        WHERE{
            {?s ?p ?o .
            ?s Property:HasUuid \"""" + str(my_uuid) + """\" .
            ?s Property:HasName ?s_label .
            ?p Property:HasName ?p_label .
            ?o Property:HasName ?o_label .}
            UNION
            {?s ?p ?o .
            ?o Property:HasUuid \"""" + str(my_uuid) + """\" .
            ?s Property:HasName ?s_label .
            ?p Property:HasName ?p_label .
            ?o Property:HasName ?o_label .
            }
            UNION
            {?s ?p ?o .
            ?s Property:HasUuid \"""" + str(my_uuid) + """\" .
            }
        } """

    print(sparql_query)
    sparql = SPARQLWrapper(sparql_url)
    sparql.setHTTPAuth('BASIC')
    sparql.setCredentials(os.environ.get("BLAZEGRAPH_USER"), os.environ.get("BLAZEGRAPH_PASSWORD"))

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


class GetTopicTaxonomyInput(BaseModel):
    osw_id: str = Field(..., description="The id of the central OSW element to find out all parent and sub-classes")
    parent_depth: int = Field(10, description="The depth of searching for the parent classes")
    child_depth: int = Field(1, description="The depth of searching for child classes")


@tool
def get_topic_taxonomy(inp: GetTopicTaxonomyInput):
    """Get all parent classes and some child classes of a given class or name.
    Can be used to find out what something is or which more special versions of that thing are described."""

    sparql_url = os.environ.get("BLAZEGRAPH_ENDPOINT")

    osw_id = inp.osw_id.split(":")[-1].split('.')[0]  ## second split: get rid of file ending
    my_uuid = str(UUID(osw_id.replace("OSW", "")))
    sparql_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
        PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
        PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
        PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
		PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>

        SELECT DISTINCT ?s_label ?p ?p_label ?o ?o_label ?s_id ?o_id
        WHERE{
            {?s ?p ?o .
            ?s (^Property:SubClassOf)*/Property:HasUuid \"""" + str(my_uuid) + """\" .
            ?s Property:HasName ?s_label .
            ?p Property:HasName ?p_label .
            ?o Property:HasName ?o_label .
            ?o Property:HasOswId ?o_id .} 
            UNION 
            {?s ?p ?o .
            ?s (Property:SubClassOf)*/Property:HasUuid \"""" + str(my_uuid) + """\" .
            ?s Property:HasName ?s_label .
            ?p Property:HasName ?p_label .
            ?o Property:HasName ?o_label .
            ?o Property:HasOswId ?o_id .}

        } """

    print(sparql_query)
    sparql = SPARQLWrapper(sparql_url)
    sparql.setHTTPAuth('BASIC')
    sparql.setCredentials(os.environ.get("BLAZEGRAPH_USER"), os.environ.get("BLAZEGRAPH_PASSWORD"))

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    print("resultate des queries", results)
    return results


class GetInstancesInput(BaseModel):
    osw_id: str = Field(..., description="The id of the category to find instances i.e. examples")
    max_number: int = Field(10, description="The maximum number of instances to be fetched")


@tool
def get_instances(inp: GetInstancesInput):
    """Get all instances of a given class.
    Can be used to find examples for something."""

    try:
        sparql_url = os.environ.get("BLAZEGRAPH_ENDPOINT")

        osw_id = inp.osw_id.split(":")[-1].split('.')[0]  ## second split: get rid of file ending
        my_uuid = str(UUID(osw_id.replace("OSW", "")))
        sparql_query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX osl: <https://mat-o-lab.open-semantic-lab.org/id/>
            PREFIX Property: <https://mat-o-lab.open-semantic-lab.org/id/Property-3A>
            PREFIX File: <https://mat-o-lab.open-semantic-lab.org/id/File-3A>
            PREFIX Category: <https://mat-o-lab.open-semantic-lab.org/id/Category-3A>
            PREFIX Item: <https://mat-o-lab.open-semantic-lab.org/id/Item-3A>

            SELECT DISTINCT ?s_label ?p ?p_label ?o ?o_label ?s_id ?o_id
            WHERE{
                {?s ?p ?o .
                ?s Property:HasType/(^Property:SubClassOf)*/Property:HasUuid \"""" + str(my_uuid) + """\" .
                ?s Property:HasName ?s_label .
                ?p Property:HasName ?p_label .
                ?o Property:HasName ?o_label .
                ?o Property:HasOswId ?o_id .} 
                UNION 
                {?s ?p ?o .
                ?s Property:HasType/(Property:SubClassOf)*/Property:HasUuid \"""" + str(my_uuid) + """\" .
                ?s Property:HasName ?s_label .
                ?p Property:HasName ?p_label .
                ?o Property:HasName ?o_label .
                ?o Property:HasOswId ?o_id .}
            } """

        # print(sparql_query)
        sparql = SPARQLWrapper(sparql_url)
        sparql.setHTTPAuth('BASIC')
        sparql.setCredentials(os.environ.get("BLAZEGRAPH_USER"), os.environ.get("BLAZEGRAPH_PASSWORD"))

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        print("resultate des queries", results)
        return results
    except Exception as e:
        return ("no instances found. The name of the class might be not avialable. Try to use the search funciton to "
                "find available classes")


class GetWebsiteHtmlInput(BaseModel):
    url: str = Field(..., description="The url of the website to get the html from.")


@tool
def get_website_html(inp:GetWebsiteHtmlInput):
    """gets the html content of a webpage."""

    url = inp.url

    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    return html

