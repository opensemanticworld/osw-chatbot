import networkx as nx
import matplotlib.pyplot as plt
from langchain_core.documents import Document
from langchain_graphrag.indexing import TextUnitExtractor
from langchain_text_splitters import CharacterTextSplitter
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipExtractor,
)
from langchain_graphrag.indexing.graph_generation import GraphsMerger
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipDescriptionSummarizer,
)
from langchain_graphrag.indexing.graph_generation import (
    GraphGenerator,
)
from osw_chatbot.llm import LLM as er_llm
from osw_chatbot.llm import LLM as es_llm


SOME_TEXT = """
Contrary to popular belief, Lorem Ipsum is not simply random text. 
It has roots in a piece of classical Latin literature from 45 BC, 
making it over 2000 years old. Richard McClintock, a Latin professor 
at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words,
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in 
classical literature, discovered the undoubtable source. Lorem Ipsum comes 
from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" 
(The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a 
treatise on the theory of ethics, very popular during the Renaissance. 
The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", 
comes from a line in section 1.10.32.

The standard chunk of Lorem Ipsum used since the 1500s is reproduced below 
for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et 
Malorum" by Cicero are also reproduced in their exact original form, accompanied
by English versions from the 1914 translation by H. Rackham.
"""  # noqa

document = Document(page_content=SOME_TEXT)
splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
text_unit_extractor = TextUnitExtractor(text_splitter=splitter)
df_text_units = text_unit_extractor.run([document])
print(df_text_units.head())

# There is a static method provide to build the default extractor
extractor = EntityRelationshipExtractor.build_default(llm=er_llm)


def analyse_graph():
    text_unit_graphs = extractor.invoke(df_text_units)
    for index, g in enumerate(text_unit_graphs):
        print("---------------------------------")
        print(f"Graph: {index}")
        print(f"Number of nodes - {len(g.nodes)}")
        print(f"Number of edges - {len(g.edges)}")
        print(g.nodes())
        print(g.edges())
        print("---------------------------------")

    # You will see that every node has `description` and `text_unit_ids` as attributes # noqa
    print(text_unit_graphs[0].nodes["RICHARD MCCLINTOCK"])
    # You will see that every edge has `weight`, `description` and `text_unit_ids` as attributes # noqa
    print(
        text_unit_graphs[0].edges[
            ("RICHARD MCCLINTOCK", "HAMPDEN-SYDNEY COLLEGE")
        ]
    )


graphs_merger = GraphsMerger()


summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=es_llm)


graph_generator = GraphGenerator(
    er_extractor=extractor,
    graphs_merger=GraphsMerger(),
    er_description_summarizer=summarizer,
)

(er_sanitized_graph, er_summarized_graph) = graph_generator.run(df_text_units)

G = er_summarized_graph

print(f"Number of nodes - {len(G.nodes)}")
print(f"Number of edges - {len(G.edges)}")


fig = plt.figure(figsize=(12, 12))
pos = nx.kamada_kawai_layout(G)
nx.draw(
    G
)  # nx.spring_layout(er_sanitized_graph), node_size=1500, node_color='yellow', font_size=8, font_weight='bold') # noqa
nx.draw_networkx_labels(G, pos)
# plt.savefig("Graph.png", format="PNG")
plt.show()
