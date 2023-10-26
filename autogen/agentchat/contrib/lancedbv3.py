

from typing import Callable, Dict, List, Optional

from overrides import override
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import get_files_from_dir, split_files_to_chunks
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
logger = logging.getLogger(__name__)

try:
    import lancedb
except ImportError as e:
    logging.fatal("lancedb is not installed. Try running 'pip install lancedb'")
    raise e        



db_path = "/tmp/lancedb"


def create_lancedb():

    db = lancedb.connect(db_path)
    embeddings = OpenAIEmbeddings(openai_api_key='s')
    table = db.create_table("my_table", data=[
            {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
        ], mode="overwrite")
        
class LancedbRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def query_vector_db(
        self,
        query_texts,
        n_results=10,
        search_string="",):

        create_lancedb()
        
        if query_texts:
            vector = [0.1, 0.3]
        db = lancedb.connect(db_path)
        table = db.open_table("my_table")
        query = table.search(vector).where(f"documents LIKE '%{search_string}%'").limit(n_results).to_df()
        data ={"ids": [query["id"].tolist()], "documents": [query["documents"].tolist()]}
        return data




    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = ""):
        results = self.query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
        )

        self._results = results
        print("doc_ids: ", results["ids"])
