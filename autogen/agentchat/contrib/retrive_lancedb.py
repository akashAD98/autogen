from typing import Callable, Dict, List, Optional

from overrides import override
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import get_files_from_dir, split_files_to_chunks
import logging

logger = logging.getLogger(__name__)

try:
    import lancedb
except ImportError as e:
    logging.fatal("lancedb is not installed. Try running 'pip install lancedb'")
    raise e        



db_path = "/tmp/lancedb"

def create_lancedb():

    db = lancedb.connect(db_path)
    data = [
                {"vector": [1.1, 1.2], "id": 11, "documents": "This is a test document spark"},
                {"vector": [0.2, 1.8], "id": 22, "documents": "This is another test document"},
                {"vector": [0.1, 0.3], "id": 3, "documents": "This is a third test document spark"},
                {"vector": [0.5, 0.7], "id": 44, "documents": "This is a fourth test document"},
                {"vector": [2.1, 1.3], "id": 55, "documents": "This is a fifth test document spark"},
                {"vector": [5.1, 8.3], "id": 66, "documents": "This is a sixth test document"},
            ]
    try:
        db.create_table("my_table", data)
    except OSError:
        pass

class LancedbRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def query_vector_db(
        self,
        query_texts,
        n_results=10,
        search_string="",):

        
        if query_texts:
            vector = [0.1, 0.3]
        db = lancedb.connect(db_path)
        table = db.open_table("my_table")
        query = table.search(vector).where(f"documents LIKE '%{search_string}%'").limit(n_results).to_df()
        data ={"ids": query["id"].tolist(), "documents": query["documents"].tolist()}
        return data




    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = ""):
        results = self.query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
        )

        self._results = results
        print("doc_ids: ", results["ids"])
