"""
Saves a research report to Google Drive using the Anthropic API
with the Google Drive MCP server.

Requires:
- ANTHROPIC_API_KEY set in your environment (.env or system)
- Google Drive connected in Claude.ai (already done)

Usage:
    from utils.drive_saver import save_report_to_drive
    result = save_report_to_drive(report_text, filename="NVIDIA_report.txt")
"""

import os
import anthropic


DRIVE_MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"


def save_report_to_drive(report: str, filename: str = "research_report.txt") -> str:
    """
    Saves the report string as a file in Google Drive.
    Returns a status message with the file link if successful.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return (
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or run: "
            "set ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            tools=[
                {
                    "type": "mcp",
                    "server_url": DRIVE_MCP_URL,
                    "server_name": "google-drive"
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Create a new Google Drive file called '{filename}' "
                        f"with the following content:\n\n{report}\n\n"
                        "After creating it, return the file name and its shareable link."
                    )
                }
            ],
            betas=["mcp-client-2025-04-04"]
        )

        # Extract text response from content blocks
        for block in response.content:
            if hasattr(block, "text") and block.text:
                return block.text

        return "File saved to Google Drive successfully."

    except anthropic.AuthenticationError:
        return "Invalid ANTHROPIC_API_KEY. Check your key and try again."

    except Exception as e:
        return f"Drive save failed: {e}"
