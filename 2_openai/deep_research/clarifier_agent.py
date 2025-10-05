# clarifier_agent.py
from pydantic import BaseModel, Field
from agents import Agent 

class ClarifyingOutput(BaseModel):
    questions: list[str] = Field(description="Three brief clarifying questions for the user's query.")
    refined_query_hint: str = Field(description="A concise refined query to improve the research.")

_INSTRUCTIONS = """ You are a helpful research assistant.
Given a user query, produce three brief clarifying questions that help refine the research,
and propose ONE refined query hint that would likely improve outcomes.
Constraints:
- 3 concise, concrete, non-overlapping questions (<= 15 words each).
- Avoid yes/no; prefer specifics (e.g., 'Which neighborhoods...' not 'Do you want...').
- refined_query_hint should be one clean query string.
"""

clarifier_agent = Agent(
    name="clarifier_agent",
    instructions=_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarifyingOutput,
)
