from dotenv import load_dotenv
import tomllib
from document_loader import DocumentLoader, VectorStore

def load_toml() -> dict:
    with open("../config.toml", "rb") as f:
        return tomllib.load(f)

load_dotenv()
config = load_toml()

embedding = config["embedding"]["model"]

file = DocumentLoader()
all_splits = file.load_and_split_file(file_path=config["upload"]["file_path"])

new_index = VectorStore(elasticsearch_url=config["elastic"]["url"],
                        elasticsearch_user=config["elastic"]["user"],
                        elasticsearch_password=config["elastic"]["password"],
                        embedding_model=embedding,
                        index_name=config["upload"]["index_name"]
                        )

new_index.create_index(documents=all_splits)
