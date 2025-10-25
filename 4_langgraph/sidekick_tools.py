from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

# Provides a Playwright-powered browser toolkit for web navigation tasks.
# Returns both tools and live browser/runtime handles to allow explicit cleanup.
# Note: ensure proper shutdown to avoid orphan Chromium processes (resource leaks).
async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright

# Thin wrapper over Pushover's REST API to send real-time push notifications.
# Requires PUSHOVER_TOKEN and PUSHOVER_USER in the environment; network failures should be handled upstream.
# Returns "success" on successful send, or raises an exception on failure.
def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"

# Exposes a constrained file-management toolkit rooted at the dir "./sandbox".
# Grants the agent read/write capabilities while sandboxing it to a safe project subdirectory.
def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    # Web search tool backed by GoogleSerperAPIWrapper.
    # Use for retrieving SERP snapshots (titles/snippets/links) — not for authenticated browsing.
    tool_search =Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    # Wikipedia lookup tool: queries the public Wikipedia API via LangChain wrapper.
    # Ideal for quick factual lookups and summaries.
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # Python REPL tool: for executing code snippets in the agent's sandboxed environment.
    # No sandboxing effect - all Python code runs on locally in a single, isolated process.
    python_repl = PythonREPLTool()
    
    return file_tools + [push_tool, tool_search, python_repl,  wiki_tool]

