from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from clarifier_agent import clarifier_agent, ClarifyingOutput
from typing import Iterable, Optional
import asyncio

ASK_USER_NOTE = (
    "❓ **Please answer the following questions** so I can give you the most focused and useful result.\n"
    "After you fill your answers (you may leave any blank), **press Run again**.\n"
)
class ResearchManager:
    """
    Orchestrates the deep-research flow with a two-run UX:
      • 1st Run: produce clarifying questions and stop.
      • 2nd Run: run full research with (optional) user answers.

    Optional email sending only happens if 'request_email' is provided.
    """

    # You can expose this for the UI to read after a run
    last_markdown_report: str = ""

    async def run(self, query: str, clarifying_answers: Optional[Iterable[str]] = None, request_email: Optional[str] = None):
        """
        Run the deep research process.

        Behavior:
          • If no (or empty) clarifying_answers are provided, we generate 3 clarifying questions,
            yield them, yield an instruction note, and STOP (user should press Run again).
          • If answers are provided, we improve the query, plan searches, perform searches,
            write the report, and (optionally) send the email if 'request_email' is given.

        Yields:
          status lines as markdown (and finally the full markdown report).
        """
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            link = f"https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print(f"View trace: {link}")
            yield f"View trace: {link}"
            # Normalize answers
            answers = list(clarifying_answers or [])
            answers_trimmed = [a.strip() for a in answers if a and a.strip()]
            has_effective_answers = len(answers_trimmed) > 0

            # If it's the first run (no answers yet) -> clarify and stop
            if not has_effective_answers:
                print("Clarify stage (first run): generating questions...")
                clar = await self.clarify_query(query)
                qs = (clar.questions or [])[:3]
                while len(qs) < 3:
                    qs.append("")  # pad to 3 (stable UI)

                # Show questions and the instruction message; STOP here
                yield "### Clarifying Questions\n" + "\n".join(
                    [f"**Q{i+1}.** {q}" for i, q in enumerate(qs) if q]
                )
                yield ASK_USER_NOTE
                return  # user will answer and press Run again

            # Otherwise: second run with answers -> improve query and continue
            print("Second run: improving query with answers...")
            clar = await self.clarify_query(query)  # get refined hint as well
            improved_query = self._build_improved_query(query, clar, answers_trimmed)
            print(f"Improved query: {improved_query}")

            # Plan searches
            print("Starting research...")
            yield "🧭 Planning searches..."
            search_plan = await self.plan_searches(improved_query)
            # Perform searches (parallel)  
            yield "🔍 Starting searches..." 
            search_results = await self.perform_searches(search_plan)
            yield "✅ Searches complete. Generating report..."
            # Write report
            report = await self.write_report(improved_query, search_results)
            self.last_markdown_report = report.markdown_report
            # Present result
            yield "📄 Report ready (see below)."
            yield report.markdown_report
            # Email (optional)
            if request_email and request_email.strip():
                yield f"✉️ Sending to {request_email.strip()}..."
                send_res = await self.send_email(report, to_email=request_email.strip())
                status = send_res.get("status", "")
                reason = send_res.get("reason", "")
                if status == "success":
                    yield "✅ Email sent."
                elif status == "skipped":
                    yield "ℹ️ Email skipped."
                else:
                    yield f"⚠️ Email not sent: {reason or 'unknown error'}"
            else:
                yield (
                    "📬 If you'd like me to email this report to you, please type your email "
                    "address and press **Send** (or press **Skip** to finish here)."
                )
            
        
    async def clarify_query(self, query: str) -> ClarifyingOutput:
        """Ask the clarifier agent to produce 3 questions + a refined query hint."""
        result = await Runner.run(
            clarifier_agent,
            [{"role": "user", "content": query}],
        )
        # Depending on your Agent runner, either result.final_output is already a ClarifyingOutput,
        # or you may need a method like 'final_output_as(ClarifyingOutput)'
        if hasattr(result, "final_output_as"):
            return result.final_output_as(ClarifyingOutput)
        return result.final_output  # type: ignore

    def _build_improved_query(self, original_query: str, clar: ClarifyingOutput, answers_trimmed: list[str]) -> str:
        """Combine refined hint and user answers to build an improved query."""
        refined = (clar.refined_query_hint or "").strip()
        base = refined if refined else original_query
        if answers_trimmed:
            base = f"{base} — user specifics: {'; '.join(answers_trimmed)}"
        return base

    async def plan_searches(self, query: str) -> WebSearchPlan:
        """ Plan the searches to perform for the query using the planner agent."""
        print("Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        print(f"Will perform {len(result.final_output.searches)} searches")
        # keep your existing helper if present:
        if hasattr(result, "final_output_as"):
            return result.final_output_as(WebSearchPlan)
        return result.final_output  

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Perform the searches to perform for the query. """
        print("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results: list[str] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> Optional[str]:
        """ Perform a search for the query using the search agent."""
        user_input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(search_agent, user_input)
            return str(result.final_output)
        except Exception:
            return None

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query using the writer agent."""
        print("Thinking about report...")
        user_input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(writer_agent, user_input)
        print("Finished writing report")
        if hasattr(result, "final_output_as"):
            return result.final_output_as(ReportData)
        return result.final_output 
    
    async def send_email(self, report: ReportData, to_email: Optional[str]) -> dict:
        """
        Send the report via the email agent only if 'to_email' is provided.
        Returns a status dict like: {"status": "success"|"error"|"skipped", "code": "...", "reason": "..."}.
        The email agent (per its instructions) will call the tool exactly once
        or return 'skipped' if no recipient is provided.
        """
        print("Writing email...")
        if not to_email or not to_email.strip():
            # mimic the email agent's skip behavior without calling it
            print("Email skipped (no recipient).")
            return {"status": "skipped", "code": "", "reason": "missing recipient email"}
        # Compose a compact prompt for the email agent
        prompt = (
            f"to_email: {to_email.strip()}\n"
            f"subject_hint: Deep Research Report\n"
            f"report_markdown:\n{report.markdown_report}"
        )
        result = await Runner.run(email_agent, prompt)
        print("Email attempted")
        try:
            return getattr(result, "tool_result", {"status": "success"})
        except Exception:
            return {"status": "success"}