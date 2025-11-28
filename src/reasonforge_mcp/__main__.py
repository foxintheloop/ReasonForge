"""
ReasonForge MCP Server - Entry Point

Run this module to start the MCP server:
    python -m reasonforge_mcp
"""

import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
