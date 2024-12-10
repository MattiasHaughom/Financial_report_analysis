from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from .llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    sections: List[dict] = Field(
        description="Company name and the sections of the report",
        default_factory=list
    )
    key_points: List[str] = Field(
        description="List of key points extracted from the documents",
        default_factory=list
    )
    sources: List[str] = Field(
        description="Document IDs used",
        default_factory=list
    )
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an an financial analyst that is tasked with analyzing key information from financial reports.
    You are presented with a some pages from a financial report 

    # Output Structure
    Your response should be organized as follows:
    1. Key Points: Bullet points of the most important information


    # Guidelines:
    
    
    Review the question from the user:
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context from Financial reports.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the financial reports.

        Returns:
            A SynthesizedResponse containing thought process, answer, and context sufficiency.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )