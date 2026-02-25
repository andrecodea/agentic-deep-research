from langchain_chroma import Chroma


chroma_client = Chroma()
collection = chroma_client.create_collection("research_agent")