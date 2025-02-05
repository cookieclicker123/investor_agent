import chainlit as cl
from src.data_model import (
    LLMResponse, WebAgentResponse, 
    FinanceAgentResponse, PDFAgentResponse
)
import json

def create_ui_components():
    """Factory function for UI components"""
    
    async def show_query_analysis(response: LLMResponse) -> None:
        """Display meta-agent query analysis with raw JSON"""
        # Format readable summary with safe access to optional fields
        summary = f"""# 🔍 Query Analysis

## Query Type & Complexity
- Type: `{getattr(response, 'query_type', 'Unknown')}`
- Complexity: `{getattr(response, 'complexity', 'Unknown')}`

## Selected Agents
{chr(10).join(f'• 🤖 {intent.value.replace("_", " ").title()}' for intent in response.intents)}

## Strategy
{response.raw_response.get('raw_text', 'No strategy available')}
"""
        # Create expandable section with raw JSON
        await cl.Message(content=summary, author="system").send()
        
        # Show raw JSON in expandable section
        json_content = json.dumps(response.dict(), indent=2)
        await cl.Message(
            content=f"""<details>
<summary>🔍 Raw Query Analysis Data</summary>

```json
{json_content}
```
</details>""",
            author="system"
        ).send()

    async def show_finance_analysis(response: FinanceAgentResponse) -> None:
        """Display finance agent analysis with raw JSON"""
        # Format readable summary
        if response.error:
            summary = f"⚠️ Finance Analysis Error: {response.error}"
        else:
            summary = "## 📈 Financial Data\n\n"
            for stock in response.stock_data:
                summary += f"""### {stock.symbol}
- Current Price: ${stock.current_price.price:.2f}
- Change: {stock.current_price.change_percent:.2f}%
- Volume: {stock.current_price.volume:,}
- Trading Day: {stock.current_price.trading_day}

**Fundamentals**
- Market Cap: {stock.fundamentals.market_cap or 'N/A'}
- P/E Ratio: {stock.fundamentals.pe_ratio or 'N/A'}
- EPS: {stock.fundamentals.eps or 'N/A'}
"""

        # Show summary
        await cl.Message(content=summary, author="system").send()
        
        # Show raw JSON in expandable section
        json_content = json.dumps(response.dict(), indent=2)
        await cl.Message(
            content=f"""<details>
<summary>📈 Raw Financial Data</summary>

```json
{json_content}
```
</details>""",
            author="system"
        ).send()

    async def show_web_analysis(response: WebAgentResponse) -> None:
        """Display web agent analysis with raw JSON"""
        # Format readable summary
        if response.error:
            summary = f"⚠️ Web Analysis Error: {response.error}"
        else:
            summary = "## 🌐 Web Search Results\n\n"
            for result in response.search_results:
                summary += f"""### [{result.title}]({result.link})
{result.snippet}
*Date: {result.date}*

"""
        # Show summary
        await cl.Message(content=summary, author="system").send()
        
        # Show raw JSON in expandable section
        json_content = json.dumps(response.dict(), indent=2)
        await cl.Message(
            content=f"""<details>
<summary>🌐 Raw Web Search Data</summary>

```json
{json_content}
```
</details>""",
            author="system"
        ).send()

    async def show_pdf_analysis(response: PDFAgentResponse) -> None:
        """Display PDF agent analysis with raw JSON"""
        # Format readable summary
        if response.error:
            summary = f"⚠️ PDF Analysis Error: {response.error}"
        else:
            summary = "## 📚 PDF Analysis Results\n\n"
            for result in response.relevant_chunks:
                summary += f"""### Document: {result.source}
{result.content}
*Relevance Score: {result.relevance_score:.2f}*

"""

        # Show summary
        await cl.Message(content=summary, author="system").send()
        
        # Show raw JSON in expandable section
        json_content = json.dumps(response.dict(), indent=2)
        await cl.Message(
            content=f"""<details>
<summary>📚 Raw PDF Analysis Data</summary>

```json
{json_content}
```
</details>""",
            author="system"
        ).send()

    async def show_synthesis_header() -> None:
        """Show synthesis phase header"""
        await cl.Message(
            content="# 🧠 Synthesizing Final Analysis...",
            author="system"
        ).send()

    return {
        "show_query_analysis": show_query_analysis,
        "show_finance_analysis": show_finance_analysis,
        "show_web_analysis": show_web_analysis,
        "show_pdf_analysis": show_pdf_analysis,
        "show_synthesis_header": show_synthesis_header
    }
