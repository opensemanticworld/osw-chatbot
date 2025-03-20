# ruff: noqa: B008
# ruff: noqa: E402
# ruff: noqa: ERA001

import logging
import os
from pathlib import Path

import typer

_LOGGER = logging.getLogger("main:indexer")

from langchain_chroma.vectorstores import Chroma as ChromaVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_graphrag.indexing import SimpleIndexer, TextUnitExtractor
from langchain_graphrag.indexing.artifacts import IndexerArtifacts
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from langchain_graphrag.indexing.graph_clustering.leiden_community_detector import (
    HierarchicalLeidenCommunityDetector,
)
from langchain_graphrag.indexing.graph_generation import (
    EntityExtractionPromptBuilder,
    EntityRelationshipDescriptionSummarizer,
    EntityRelationshipExtractor,
    GraphGenerator,
    GraphsMerger,
)
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)
from langchain_text_splitters import TokenTextSplitter

from osw_chatbot.graphrag.common import (
    EmbeddingModelType,
    LLMType,
    get_artifacts_dir_name,
    load_artifacts,
    make_embedding_instance,
    make_llm_instance,
    save_artifacts,
    trace_via_langsmith,
)

# app = Typer()


# @app.command()
def index(
    input_file: Path = Path(
        "input-data", "book.txt"
    ),  # = typer.Option(..., dir_okay=False, file_okay=True, exists=True),
    output_dir: Path = Path(
        "temp"
    ),  # = typer.Option(..., dir_okay=True, file_okay=False),
    cache_dir: Path = Path(
        "temp", "cache"
    ),  # = typer.Option(..., dir_okay=True, file_okay=False),
    llm_type: LLMType = LLMType.azure_openai,  # = typer.Option(..., case_sensitive=False),
    llm_model: str = "gpt-4o-2024-08-06",  # = typer.Option(..., case_sensitive=False),
    embedding_type: EmbeddingModelType = EmbeddingModelType.azure_openai,  # = typer.Option(..., case_sensitive=False),
    embedding_model: str = "text-embedding-ada-002-2",  # = typer.Option(..., case_sensitive=False),
    chunk_size: int = 1200,  # = typer.Option(1200, help="Chunk size for text splitting"),
    chunk_overlap: int = 100,  # = typer.Option(100, help="Chunk overlap for text splitting"),
    ollama_num_context: int = None,  # = typer.Option(None, help="Context window size for ollama model"),
    enable_langsmith: bool = False,  # = typer.Option(False, help="Enable Langsmith"),  # noqa: FBT001, FBT003
) -> IndexerArtifacts:
    if enable_langsmith:
        trace_via_langsmith()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir = output_dir / "vector_stores"
    artifacts_dir = output_dir / get_artifacts_dir_name(llm_model)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # tableprint.table(
    print(
        [
            ["LangSmith", str(enable_langsmith)],
            ["Input file", str(input_file)],
            ["Cache directory", str(cache_dir)],
            ["Vector store directory", str(vector_store_dir)],
            ["Artifacts directory", str(artifacts_dir)],
            ["LLM Type", llm_type],
            ["LLM Model", llm_model],
            ["Embedding Type", embedding_type],
            ["Embedding Model", embedding_model],
            ["Chunk Size", chunk_size],
            ["Chunk Overlap", chunk_overlap],
            ["OLLAMA_HOST", os.getenv("OLLAMA_HOST")],
            [
                "Ollama Num Context",
                "Not Provided"
                if ollama_num_context is None
                else ollama_num_context,
            ],
        ]
    )

    ######### Start of creation of various objects/dependencies #############

    # Dataloader that loads the supplied text file for indexing
    documents = TextLoader(file_path=input_file, encoding="utf-8").load()

    # TextSplitter required by TextUnitExtractor
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # TextUnitExtractor that extracts text units from the text files
    text_unit_extractor = TextUnitExtractor(text_splitter=text_splitter)

    # Entity Relationship Extractor
    # entity_extractor = EntityRelationshipExtractor.build_default(
    #     llm=make_llm_instance(llm_type, llm_model, cache_dir),
    #     chain_config={"tags": ["er-extraction"]},
    #
    # )
    entity_extractor = EntityRelationshipExtractor(
        prompt_builder=EntityExtractionPromptBuilder(
            entity_types=["person", "organization", "location", "date"]
        ),
        llm=make_llm_instance(llm_type, llm_model, cache_dir),
        chain_config={"tags": ["er-extraction"]},
    )

    # Entity Relationship Description Summarizer
    entity_summarizer = EntityRelationshipDescriptionSummarizer.build_default(
        llm=make_llm_instance(llm_type, llm_model, cache_dir),
        chain_config={"tags": ["er-description-summarization"]},
    )

    # Graph Generator
    graph_generator = GraphGenerator(
        er_extractor=entity_extractor,
        graphs_merger=GraphsMerger(),
        er_description_summarizer=entity_summarizer,
    )

    # Community Detector
    community_detector = HierarchicalLeidenCommunityDetector()

    # Entities artifacts Generator
    # We need the vector Store (mandatory) for entities

    # let's create a collection name based on
    # the embedding model name
    entities_collection_name = f"entity-{embedding_model}"
    entities_vector_store = ChromaVectorStore(
        collection_name=entities_collection_name,
        persist_directory=str(vector_store_dir),
        embedding_function=make_embedding_instance(
            embedding_type=embedding_type,
            model=embedding_model,
            cache_dir=cache_dir,
        ),
    )

    entities_artifacts_generator = EntitiesArtifactsGenerator(
        entities_vector_store=entities_vector_store
    )

    relationships_artifacts_generator = RelationshipsArtifactsGenerator()

    # Community Report Generator
    report_gen_llm = make_llm_instance(llm_type, llm_model, cache_dir)
    report_generator = CommunityReportGenerator.build_default(
        llm=report_gen_llm,
        chain_config={"tags": ["community-report"]},
    )

    report_writer = CommunityReportWriter()

    communities_report_artifacts_generator = (
        CommunitiesReportsArtifactsGenerator(
            report_generator=report_generator,
            report_writer=report_writer,
        )
    )

    text_units_artifacts_generator = TextUnitsArtifactsGenerator()

    ######### End of creation of various objects/dependencies #############

    indexer = SimpleIndexer(
        text_unit_extractor=text_unit_extractor,
        graph_generator=graph_generator,
        community_detector=community_detector,
        entities_artifacts_generator=entities_artifacts_generator,
        relationships_artifacts_generator=relationships_artifacts_generator,
        text_units_artifacts_generator=text_units_artifacts_generator,
        communities_report_artifacts_generator=communities_report_artifacts_generator,
    )

    print("Generate Artifacts")
    artifacts = indexer.run(documents)
    print("Artifacts generated")
    # save the artifacts
    save_artifacts(artifacts, artifacts_dir)
    artifacts.report()
    return artifacts


# @app.command()
def report(
    artifacts_dir: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
):
    _LOGGER.info("Artifacts directory - %s", artifacts_dir)

    artifacts: IndexerArtifacts = load_artifacts(artifacts_dir)
    artifacts.report()
