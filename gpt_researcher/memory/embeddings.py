from langchain_community.vectorstores import FAISS
import os


class Memory:
    def __init__(self, embedding_provider, cfg = None, **kwargs):

        _embeddings = None
        match embedding_provider:
            case "ollama":
                from langchain.embeddings import OllamaEmbeddings
                _embeddings = OllamaEmbeddings(model="llama2")
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(
                    openai_api_key=cfg.openai_api_key if hasattr(cfg,'openai_api_key') else None,
                    openai_api_base=cfg.openai_api_base if hasattr(cfg,'openai_api_base') else None,
                    model=cfg.embedding_model if hasattr(cfg,'embedding_model') else 'text-embedding-ada-002',
                )
            case "azureopenai":
                from langchain_openai import AzureOpenAIEmbeddings
                _embeddings = AzureOpenAIEmbeddings(deployment=os.environ["AZURE_EMBEDDING_MODEL"], chunk_size=16)
            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings
                _embeddings = HuggingFaceEmbeddings()

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
