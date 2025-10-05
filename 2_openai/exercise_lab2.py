from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from typing import Dict, Any
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio
import json
import time

load_dotenv(override=True)

# ===Tools for sale manager agent===

instructions_sales_agent1 = "You are a sales agent working for Nayax, \
 a global commerce and payments platform. You help merchants grow with \
 end-to-end cashless acceptance (80+ payment methods, 50+ currencies), \
 management tools, and consumer engagement. You write professional, \
 serious cold emails tailored to Nayax's value."

instructions_sales_agent2 = "You are a humorous, engaging sales agent working for Nayax, \
 a global commerce and payments platform. You write witty, engaging cold \
 emails that highlight Nayax's global footprint (120+ countries) and \
 consumer engagement tools to drive response."

instructions_sales_agent3 = "You are a busy sales agent working for Nayax, \
 focused on concise, to-the-point cold emails that emphasize quick \
 deployment, local cashless acceptance, and real-time business insights."

sales_agent1 = Agent(
        name="Professional Sales Agent",
        instructions=instructions_sales_agent1,
        model="gpt-4o-mini",
)

sales_agent2 = Agent(
        name="Engaging Sales Agent",
        instructions=instructions_sales_agent2,
        model="gpt-4o-mini",
)

sales_agent3 = Agent(
        name="Busy Sales Agent",
        instructions=instructions_sales_agent3,
        model="gpt-4o-mini",
)

description = "Write a cold sales email"
tool_sales_agent1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool_sales_agent2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool_sales_agent3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

instructions_sales_picker = "You pick the best Nayax cold sales email from the given options. \
 Imagine you are a merchant looking to grow with cashless acceptance and engagement; \
 pick the one you are most likely to respond to. Do not give an explanation; \
 reply with the selected email only."

sales_picker = Agent(
    name="sales_picker",
    instructions=instructions_sales_picker,
    model="gpt-4o-mini",
)

tool_sales_picker = sales_picker.as_tool(tool_name="sales_picker", tool_description="Select the best email from options")

@function_tool(strict_mode=False)
def personalization_tool(prospect: Dict[str, Any] | None = None,
                         id: str | None = None,
                         name: str | None = None,
                         company: str | None = None,
                         email: str | None = None,
                         role: str | None = None,
                         industry: str | None = None,
                         pain_points: str | None = None) -> Dict[str, Any]:
    """
    Build a short personalization snippet and variables from a prospect record.
    Required keys: name, company, role (optional: industry, pain_points)
    """
    data = prospect or {}
    # allow flat args to override
    if id is not None:
        data["id"] = id
    if name is not None:
        data["name"] = name
    if company is not None:
        data["company"] = company
    if email is not None:
        data["email"] = email
    if role is not None:
        data["role"] = role
    if industry is not None:
        data["industry"] = industry
    if pain_points is not None:
        data["pain_points"] = pain_points

    name = (data.get("name") or "there")
    company = (data.get("company") or "your company")
    role = (data.get("role") or "")
    industry = (data.get("industry") or "")
    pain = (data.get("pain_points") or "")
    snippet = (
        f"{name} at {company}" + (f" ({role})" if role else "") +
        (f". In {industry}, teams often face: {pain}." if industry or pain else ".")
    )
    return {"snippet": snippet, "vars": {"name": name, "company": company, "role": role}}

LOG_PATH = "analytics_log.jsonl"
_LAST_EVENT_CACHE: dict[str, tuple[str, float]] = {}

@function_tool(strict_mode=False)
def analytics_log(stage: str, prospect_id: str, meta: Any | None = None) -> str:
    """
    Append an analytics event to a local JSONL file.
    Required: stage (e.g., drafted|selected|sent|followup_sent), prospect_id
    Optional: meta (dict)
    """
    now = time.time()
    last = _LAST_EVENT_CACHE.get(prospect_id)
    if stage == "handed_off" and last and last[0] == stage and (now - last[1]) < 2.0:
        return "skipped-duplicate"

    event: Dict[str, Any] = {"stage": stage, "prospect_id": prospect_id, "ts": now}
    if meta is not None:
        # Coerce non-dict meta into a dict to avoid schema validation issues
        if isinstance(meta, dict):
            event["meta"] = meta
        else:
            event["meta"] = {"info": meta}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    _LAST_EVENT_CACHE[prospect_id] = (stage, now)
    return "ok"

tools_for_sale_manager = [tool_sales_agent1, tool_sales_agent2, tool_sales_agent3, 
                          tool_sales_picker, personalization_tool, analytics_log]

# ===Tools for email manager agent===

subject_instructions = "You can write a subject for a Nayax cold sales email. \
 You are given a message and you need to write a subject that highlights Nayax's value (global cashless acceptance, engagement tools, business management) and is likely to get a response."

html_instructions = "You convert a text email body to a beautiful HTML email for Nayax. \
 Input is plain text or light markdown. Output must be a complete HTML document: \
 <!DOCTYPE html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> \
 <style>Use a clean font, readable line-height, a centered container (max-width ~640px), a primary CTA button, and responsive mobile-safe styles. \
 h1/h2 prominent; paragraphs spaced; button styled with background-color #1a73e8, white text, padding, border-radius. \
 Signature must be: \"Best regards,\\nYuval Zaafrani\" (no placeholders).</style></head><body> \
 Wrap content in a centered card with light shadow/border and adequate padding; ensure valid HTML. \
 Do not include [Your Name] placeholders; always end with a signature: Best regards,<br/>Yuval Zaafrani. \
 </body></html>"

subject_writer = Agent(
            name="Email subject writer",
            instructions=subject_instructions,
            model="gpt-4o-mini")

html_converter = Agent(
            name="HTML email body converter", 
            instructions=html_instructions, 
            model="gpt-4o-mini")

html_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

@function_tool
def send_html_email(subject: str, html_body: str, to_email: str, prospect_id: str = "") -> Dict[str, str]:
    """Send an HTML email with the given subject and body to a single target recipient."""
    api_key = os.environ.get('SENDGRID_API_KEY')
    if not api_key:
        return {"status": "skipped", "code": "", "reason": "missing SENDGRID_API_KEY"}
    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    from_addr = os.environ.get("SENDGRID_FROM", "zayuvalza@gmail.com")
    from_email = Email(from_addr)
    to_email_obj = To(to_email)
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email_obj, subject, content).get()
    resp = sg.client.mail.send.post(request_body=mail)
    status_code = getattr(resp, "status_code", None)

    if status_code and int(status_code) >= 400:
        return {"status": "error", "code": str(status_code), "reason": "sendgrid_error"}
    return {"status": "success", "code": str(status_code) if status_code else ""}

tools_for_email_manager = [html_tool, subject_tool, send_html_email, analytics_log]

# ===email manager agent===

emailer_agent_instructions = """
You are an Email Formatter and Sender for Nayax. 
You receive a JSON-like input: {"body": <text>, "prospect": {"id": "...", "email": "..."}}.

Follow these steps carefully:

1. Write Subject:
   - Call subject_writer with this JSON exactly: {"input": body} to generate a compelling subject line.
   - Ensure the subject is concise, relevant, and likely to get a response.

2. Convert to HTML:
   - Call html_converter with this JSON exactly: {"input": body}. The output must be a complete, styled HTML email as per the template above and must end with "Best regards,<br/>Yuval Zaafrani".
   - Preserve formatting, clarity, and readability. Remove unnecessary Markdown if present.

3. Send Email:
   - Call send_html_email with this JSON exactly: {"subject": subject, "html_body": html_body, "to_email": prospect.email, "prospect_id": prospect.id} to deliver the email.
   - If send_html_email returns status="success" log stage="sent". If it returns status="error" or "skipped" log stage="error" with meta={"reason": reason, "code": code}.
   - Make sure to use the verified sender address (configured in the environment).

4. Log Analytics:
   - If the send result is success, call analytics_log with this JSON exactly: {"stage": "sent", "prospect_id": prospect.id, "meta": {"subject": subject, "to_email": prospect.email}}
   - If the send result is error or skipped, call analytics_log with this JSON exactly: {"stage": "error", "prospect_id": prospect.id, "meta": {"reason": reason, "code": code}}

Crucial Rules:
- Do not write subjects or HTML directly — always use the provided tools.
- Always send exactly one email per request.
- Always log the sending action with analytics_log.
- When finished, reply with DONE only.
- Do not log a "handed_off" stage from this agent.
"""

emailer_agent = Agent(
    name="emailer_agent",
    instructions=emailer_agent_instructions,
    tools=tools_for_email_manager,
    handoff_description="Convert an email to HTML and send it",
    model="gpt-4o-mini"
)

# ===sales manager agent===

sales_manager_instructions = """
You are Yuval Zaafrani, a Sales Manager at Nayax. 
Your goal is to produce and select the single best Nayax cold sales email draft for a given prospect.

Input format:
- You receive a JSON-like input: {"prompt": <goal or instructions>, "prospect": {"id": "...", "name": "...", "company": "...", "email": "...", "role": "...", "industry": "...", "pain_points": "..."}}.

Follow these steps carefully:

1. Personalization:
   - Call personalization_tool with this JSON exactly: {"prospect": prospect}
   - Incorporate the returned snippet/vars into the prompts you give the sales_agent tools.

2. Generate Drafts:
   - Call each sales_agent tool (sales_agent1/2/3) once with this JSON exactly: {"input": <prompt that includes the personalization snippet and variables>}.
   - Log each draft with analytics_log(stage="drafted", prospect_id=prospect.id, meta={which_agent}).
   - Do not proceed until all three drafts are ready.

3. Evaluate and Select:
   - Call sales_picker with this JSON exactly: {"input": <a string that contains all three drafts clearly labeled>} to select exactly one winning draft.
   - Log selection with analytics_log(stage="selected", prospect_id=prospect.id, meta={chosen_agent or short_reason}).

4. Handoff for Sending:
   - Immediately call analytics_log with this JSON exactly: {"stage": "handed_off", "prospect_id": prospect.id}.
   - Handoff the winning draft ONLY to the 'Email Manager' agent (emailer_agent) with {"body": <winning_draft>, "prospect": prospect}.

Crucial Rules:
- Never write drafts yourself — always use the sales_agent tools.
- Always use the sales_picker to choose the winner.
- Handoff exactly ONE email to the Email Manager — never more than one.
- Always log each stage: drafted, selected, handed_off.
- Call each draft tool (sales_agent1/2/3) at most once; call sales_picker exactly once.
- When finished, reply with DONE only.
"""

sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools_for_sale_manager,
    handoffs=[emailer_agent],
    model="gpt-4o-mini"
)

# single_prospect_message = {
#     "prompt": "Write a Nayax cold email about growing revenue with global cashless acceptance, consumer engagement, and business management tools.",
#     "prospect": {
#         "id": "p1",
#         "name": "Yuval",
#         "company": "Nayax",
#         "email": "zayuvalza@gmail.com",
#         "role": "Sales Leader",
#         "industry": "Payments/Commerce",
#         "pain_points": "Expanding cashless acceptance and consumer engagement in multiple regions"
#     }
# }

single_prospect_message = {
    "prompt": "Write a Nayax cold email about growing revenue with global cashless acceptance, consumer engagement, and business management tools.",
    "prospect": {
        "id": "p1",
        "name": "Lidan Katzav",
        "company": "Planview",
        "email": "lidankatzav5@gmail.com",
        "role": "QA Engineer",
        "industry": "Software/SaaS (Strategic Portfolio Management & Digital Product Development)",
        "pain_points": "Maintaining high QA coverage across multi-team releases, aligning test priorities with strategic roadmaps, reducing regression risk while accelerating value delivery, and adapting quickly to changing requirements"
    }
}

async def main() -> None:
    with trace("Exercise Lab 2"):
        result = await Runner.run(
            sales_manager,
            [{"role": "user", "content": json.dumps(single_prospect_message)}],
            max_turns=30,
        )

if __name__ == "__main__":
    asyncio.run(main())

