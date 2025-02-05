from typing import Dict, Any
import json

def create_response_formatters():
    """Factory function for response formatting utilities"""
    
    def format_json_dropdown(title: str, icon: str, data: Dict[str, Any]) -> str:
        """Format data as an HTML dropdown with JSON content"""
        return f"""<details>
<summary>{icon} {title}</summary>

```json
{json.dumps(data, indent=2)}
```
</details>"""

    def format_agent_step(icon: str, agent_name: str, content: str) -> str:
        """Format an agent processing step"""
        return f"""## {icon} {agent_name} Analysis
{content}"""

    def format_error_message(error: str) -> str:
        """Format error messages consistently"""
        return f"""⚠️ **Error**
```
{error}
```"""

    def format_synthesis_message(content: str) -> str:
        """Format the final synthesis message"""
        return f"""# 🧠 Final Analysis

{content}"""

    return {
        "format_json_dropdown": format_json_dropdown,
        "format_agent_step": format_agent_step,
        "format_error_message": format_error_message,
        "format_synthesis_message": format_synthesis_message
    }
