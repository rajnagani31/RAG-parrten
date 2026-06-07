from llm_adapter import generate_query
from vector_engin import QdrantVectorService
import asyncio

async def get_query(user_prompt: str):
    return await generate_query(user_prompt)


async def main():
    # user_prompt = "What is FastAPI?"
    user_prompt = input("Enter your question: ")
    queries = await get_query(user_prompt)
    print("Generated Queries:")
    for i, q in enumerate(queries): # type: ignore
        print(f"Query {i + 1}: {q}")

    # Example of using vector service to search with generated queries
    vector_service = QdrantVectorService()
    for i, query in enumerate(queries): # type: ignore
        results = vector_service.search(user_id=1, query=query)
        print(f"Results for Query {i + 1} -> '{query}': {results}")

if __name__ == "__main__":
    asyncio.run(main())