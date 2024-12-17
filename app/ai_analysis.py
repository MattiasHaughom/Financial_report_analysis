import os
import asyncio
import logging
from datetime import datetime
import pandas as pd
from typing import List, Dict
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from timescale_vector.client import uuid_from_time
from .services.utils import send_email
from .database.vector_store import VectorStore
from .services.synthesizer import AnalysisDependencies, AnalysisResult


vector_store = VectorStore()

# Ensure the analysis table is created
def setup_database():
    vector_store.create_analysis_table()
    vector_store.create_analysis_keyword_search_index()
    vector_store.create_reports_keyword_search_index()

setup_database()

# Define the analysis agent
analysis_agent = Agent(
    model='openai:gpt-4o-mini',  # Specify your AI model
    deps_type=AnalysisDependencies,
    result_type=AnalysisResult
    #retries=2
)

def get_default_metrics() -> Dict[str, List[str]]:
    """Return default financial and operational metrics."""
    return {
        "financial_metrics": [
            "Revenue Growth",
            "Operating Margin",
            "Net Profit Margin",
            "Return on Investment (ROI)",
            "Debt-to-Equity Ratio (D/E)"
        ],
        "operational_metrics": [
            "Market Share",
            "Employee Turnover",
            "Customer Satisfaction",
            "Operational Efficiency"
        ]
    }

@analysis_agent.system_prompt
async def construct_system_prompt(ctx: RunContext[AnalysisDependencies]) -> str:
    """Construct the system prompt for the AI model."""
    financial_metrics = ', '.join(ctx.deps.sector_metrics.get('financial_metrics', []))
    operational_metrics = ', '.join(ctx.deps.sector_metrics.get('operational_metrics', []))
    additional_context = ctx.deps.additional_context

    return f"""
You are a financial analyst specializing in the {ctx.deps.sector} sector.
Your task is to analyze a financial report for company {ctx.deps.company_name}.

Focus on the following financial metrics (if available):
{financial_metrics}

Focus on the following operational metrics (if available):
{operational_metrics}

Here are some insights from previous analyses:
{additional_context}

Provide a comprehensive analysis based on these metrics.
If certain metrics are not available in the report, acknowledge their absence and proceed with the available data.
"""

async def analyze_reports():
    """Analyze financial reports and save the results."""
    doc_ids_to_analyze = vector_store.get_reports_to_analyze()

    for doc_id in doc_ids_to_analyze:
        try:
            report_chunks = vector_store.get_report_chunks(doc_id)
            if not report_chunks:
                logging.warning(f"No report chunks found for doc_id: {doc_id}")
                continue

            company_id = report_chunks[0]['metadata']['issuerSign']
            company_data = vector_store.get_company_data(company_id)
            if not company_data:
                logging.warning(f"No company data found for company symbol: {company_id}")
                continue

            company_name = company_data['name']
            sector = company_data['sector']
            sector_metrics = vector_store.get_sector_metrics(sector) or get_default_metrics()

            query_text = ' '.join(sector_metrics.get('financial_metrics', []) + sector_metrics.get('operational_metrics', []))
            query_text += f" Extract key insights from the financial report of {company_name}."

            report_content = vector_store.hybrid_search(
                query=query_text,
                keyword_k=15,
                semantic_k=15,
                rerank=False,
                top_n=10,
                table_name='reports',
                metadata_filter={'doc_id': doc_id}
            )

            report_content_text = '\n'.join(
                item['content'].text if hasattr(item['content'], 'text') else item['content']
                for item in report_content.to_dict('records')
            )

            previous_analyses = vector_store.hybrid_search(
                query=query_text,
                keyword_k=10,
                semantic_k=10,
                rerank=False,
                top_n=5,
                table_name='analysis',
                metadata_filter={'company_id': company_id}
            )

            additional_context_text = '\n'.join(
                item['content'].text if hasattr(item['content'], 'text') else item['content']
                for item in previous_analyses.to_dict('records')
            )

            deps = AnalysisDependencies(
                company_name=company_name,
                sector=sector,
                sector_metrics=sector_metrics,
                additional_context=additional_context_text
            )

            result = await analysis_agent.run(
                user_prompt=report_content_text,
                deps=deps
            )

            analysis_record = {
                'id': str(uuid_from_time(datetime.now())),
                'metadata': {
                    'issuerSign': company_id,
                    'sector': sector,
                    'industry': company_data.get('industry'),
                    'doc_id': doc_id,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                },
                'contents': result.data.analysis_text,
                'embedding': vector_store.get_embedding(result.data.analysis_text)
            }

            send_email(
                subject=f"Financial Analysis Summaries for {company_name}",
                body=result.data.analysis_text,
                to_email="mattias.haughom@gmail.com"
            )

            vector_store.upsert_analysis(pd.DataFrame([analysis_record]))
            logging.info(f"Analysis saved for report ID {doc_id}")

        except Exception as e:
            logging.error(f"Error processing doc_id {doc_id}: {e}")

if __name__ == '__main__':
    asyncio.run(analyze_reports())