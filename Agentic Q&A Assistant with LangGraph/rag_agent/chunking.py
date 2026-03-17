from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Node


def chunk_documents(
    documents: List[Document],
    doc_id: str,
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    ) -> List[Node]:

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        node.metadata["doc_id"] = doc_id
        node.metadata["source"] = source_name

    return nodes

