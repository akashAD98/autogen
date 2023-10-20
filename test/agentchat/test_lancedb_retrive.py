import importlib
import logging
import os

import pytest

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen import ChatCompletion, config_list_from_json
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST

try:
    #from qdrant_client import QdrantClient
    import lancedb
    from autogen.agentchat.contrib.retrive_lancedb import (
        create_lancedb,
        LancedbRetrieveUserProxyAgent,
        query_vector_db,
    )


    LANCEDB_INSTALLED = True
except ImportError:
    LANCEDB_INSTALLED = False

test_dir = os.path.join(os.path.dirname(__file__), "..", "test_files")



ragragproxyagent = LancedbRetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    retrieve_config={
        "task": "qa",
        "chunk_token_size": 2000,
        "client": "__",
        "embedding_model": "all-mpnet-base-v2",
    },
)


create_lancedb()
ragragproxyagent.retrieve_docs("This is a test document spark", n_results=10, search_string="spark")
assert ragragproxyagent._results["ids"] == [3, 1, 5]


