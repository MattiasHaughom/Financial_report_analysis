from typing import List, Dict, Optional
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field


# Dependencies passed to the Agent
class AnalysisDependencies:
    def __init__(
        self,
        company_name: str,
        sector: str,
        sector_metrics: Dict[str, List[str]],
        additional_context: str
    ):
        self.company_name = company_name
        self.sector = sector
        self.sector_metrics = sector_metrics
        self.additional_context = additional_context


# Expected structure of the AI's response
class AnalysisResult(BaseModel):
    analysis_text: str = Field(description='The AI-generated analysis of the report')
    key_points: Optional[List[str]] = Field(description='List of key points identified in the report')
    metrics_evaluated: Optional[Dict[str, List[str]]] = Field(
        description='Metrics evaluated during analysis',
        example={
            'financial': ['Revenue Growth', 'Operating Margin'],
            'operational': ['Market Share', 'Customer Satisfaction']
        }
    )
