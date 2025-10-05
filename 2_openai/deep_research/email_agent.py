import os
from typing import Dict

import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool

def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return v if v is not None else default

@function_tool
def send_email(subject: str, html_body: str, to_email: str | None = None) -> Dict[str, str]:
    """ 
    Send an email with the given subject and HTML body. 
    Requires SENDGRID_API_KEY and SENDGRID_FROM in environment variables
    'to_email' MUST be provided by the user; if missing skipped (do NOT fallback).
    Returns: {"status": "success"|"error"|"skipped", "code": "...", "reason": "..."}
    """
    api_key = _env("SENDGRID_API_KEY")
    if not api_key:
        return {"status": "skipped", "code": "", "reason": "missing SENDGRID_API_KEY"}

    from_addr = _env("SENDGRID_FROM")
    if not from_addr:
        return {"status": "skipped", "code": "", "reason": "missing SENDGRID_FROM"}

    if not to_email:
        return {"status": "skipped", "code": "", "reason": "missing recipient email (user didn't provide)"}

    try:    
        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        from_email = Email(from_addr) 
        to_emails = To(to_email) 
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_emails, subject, content).get()
        response = sg.client.mail.send.post(request_body=mail)
        code = getattr(response, "status_code", None)
        if code and int(code) >= 400:
            return {"status": "error", "code": str(code), "reason": "sendgrid_error"}
        return {"status": "success", "code": str(code) if code else "", "reason": ""}
    except Exception as e:
        return {"status": "error", "code": "", "reason": str(e)}

INSTRUCTIONS = """You are an email-sending agent.

You will be given:
1) A final report (Markdown or already-HTML).
2) A recipient email address `to_email` (provided by the user).
3) Optionally a subject hint.

Your job:
- Generate a concise, descriptive Subject (<= 80 chars).
- If the report is Markdown, convert it into clean, well-presented HTML (basic inline CSS is OK).
- Call the tool `send_email(subject, html_body, to_email)` exactly once to send the email.
- If `to_email` is missing or empty: DO NOT call the tool. Instead, return the word `skipped`.

Hard rules:
- Never send to any default or placeholder address.
- No test emails, no additional recipients.
- Your final output must be either a single tool call (send_email) or the string `skipped`.
"""

email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gpt-4o-mini",
)
