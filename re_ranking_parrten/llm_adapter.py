# llm adapter for re-ranking partern

from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import os

import asyncio

from agents import Agent, Runner # type: ignore

load_dotenv()  # Load environment variables from .env file

SYSTEM_PROMPT = """
    You are a query rewriting and retrieval assistant.

    Your job is to convert a user's natural language question into multiple high-quality search queries that can be used for retrieval, vector search, RAG, hybrid search, or re-ranking systems.

    Rules:
    1. Understand the user's intent first.
    2. Generate 3-4 diverse search queries.
    3. Each query should express the same intent using different wording.
    4. Include:
    - Direct query
    - Expanded query
    - Technical/specific query
    - Alternative phrasing
    5. Do not answer the user's question.
    6. Do not explain anything.
    7. Return only the generated queries.

    Output Format:

    [
    "query 1",
    "query 2",
    "query 3",
    ]
"""
query_writer_agent = Agent(
    name="QueryRewriter",
    instructions=SYSTEM_PROMPT,
    model="gpt-4.1-mini",
)

import json
async def generate_query(user_prompt : str) -> None:
    result = await Runner.run(query_writer_agent, user_prompt)
    output = (result.final_output) # type: ignore
        
    queries = json.loads(output) # type: ignore
    queries.insert(0, user_prompt) # add original query for test re-ranking pattern
    return queries


