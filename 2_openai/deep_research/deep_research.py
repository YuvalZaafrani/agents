from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
from agents import Runner 
from email_agent import email_agent

load_dotenv(override=True)

PH_IDLE = "idle"                    # Primary state: no clarifying questions asked yet
PH_WAIT_ANS = "awaiting_answers"    # Second state: waiting for user answers to clarifying questions
PH_WAIT_EMAIL = "awaiting_email"    # Third state: report ready and waiting for email (Send/Skip)

# Load CSS from file (relative to this file)
STYLE_CSS = Path(__file__).with_name("style.css").read_text(encoding="utf-8")

def _build_outputs(
    status: str,
    report: str,
    phase: str,
    mgr: ResearchManager,
    q1_text: str | None = None,
    q2_text: str | None = None,
    q3_text: str | None = None,
    clar_open: bool = False,
    clar_visible= False,
    run_top_visible: bool = True,
    run_after_visible: bool = False,
    email_visible: bool = False,
    email_status_text: str | None = None,
    email_status_visible: bool = False,
):
    """
    Build a single tuple matching the outputs signature of the Run callbacks.
    """
    q1_upd = gr.update(value=q1_text or "", visible=bool(q1_text))
    q2_upd = gr.update(value=q2_text or "", visible=bool(q2_text))
    q3_upd = gr.update(value=q3_text or "", visible=bool(q3_text))
    clar_upd = gr.update(open=clar_open, visible=clar_visible)
    email_upd = gr.update(visible=email_visible)
    run_top_upd = gr.update(visible=run_top_visible)
    run_after_upd = gr.update(visible=run_after_visible)
    email_status_upd = gr.update(value=email_status_text or "", visible=email_status_visible)
    return (
        status, report, phase, mgr,
        q1_upd, q2_upd, q3_upd, clar_upd, email_upd,
        run_top_upd, run_after_upd, email_status_upd
    )

async def on_run(
    query: str,
    phase: str,
    ans1: str,
    ans2: str,
    ans3: str,
    manager: ResearchManager | None,
):
    """
    Single-button UX with two phases.

    Phase 1 (PH_IDLE):
      - Produce 3 clarifying questions.
      - Show instruction in Status.
      - Reveal the clarifying accordion (with Q above each answer).
      - Hide the TOP Run button; show the AFTER-NOTE Run button under the note.

    Phase 2 (PH_WAIT_ANS):
      - Run full research with (possibly blank) answers.
      - While running: show ONLY the current status message (replace, don't append),
        HIDE clarifying accordion, HIDE both Run buttons.
      - When done: show the final report and REVEAL the email row.
    """
    mgr = manager or ResearchManager()

    if not (query or "").strip():
        yield _build_outputs(
            status="⚠️ Please enter a topic to research.",
            report="",
            phase=PH_IDLE,
            mgr=mgr,
            clar_open=False,
            clar_visible=False,
            email_visible=False,
            run_top_visible=True,
            run_after_visible=False,
        )
        return

    # ---- Phase 1: first click -> show questions and move Run below the note ----
    if phase == PH_IDLE:
        # Get clarifying questions directly (so we can place each question above its answer field)
        clar = await mgr.clarify_query(query)
        qs = (clar.questions or [])[:3] + ["", "", ""]
        qs = qs[:3] 
        instruction = (
            "❓ **Please answer the following questions** so I can give you the most focused and useful result.\n"
            "After you fill your answers (you may leave any blank), **press Run again**."
        )
        # Show questions above their answer inputs; open+show the clarifying accordion.
        # Hide the top Run button and show the Run button after the note.
        yield _build_outputs(
            status=instruction,
            report="",
            phase=PH_WAIT_ANS,
            mgr=mgr,
            q1_text=f"**Q1.** {qs[0]}" if qs[0] else None,
            q2_text=f"**Q2.** {qs[1]}" if qs[1] else None,
            q3_text=f"**Q3.** {qs[2]}" if qs[2] else None,
            clar_open=True,
            clar_visible=True,          # visible ONLY after first Run
            email_visible=False,
            run_top_visible=False,      # hide top run
            run_after_visible=True,     # show run under the note
        )
        return

    # ---- Phase 2: second click -> run full flow ----
    if phase == PH_WAIT_ANS:
        answers = [ans1 or "", ans2 or "", ans3 or ""]
        
        # Stream ephemeral status: show only the current message (no accumulation).
        async for chunk in mgr.run(query=query, clarifying_answers=answers, request_email=None):
            current_status = str(chunk)
            yield _build_outputs(
                status=current_status,          # replace previous status
                report="",
                phase=PH_WAIT_EMAIL,            # switch phase
                mgr=mgr,
                # hide clarifying section during processing
                clar_open=False,
                clar_visible=False,
                email_visible=False,
                run_top_visible=False,          # hide both Run buttons while running
                run_after_visible=False,
            )

        # Done: set final report, keep statuses clean, and reveal email row
        final_report = getattr(mgr, "last_markdown_report", "")
        yield _build_outputs(
            status="",
            report=final_report,
            phase=PH_WAIT_EMAIL,
            mgr=mgr,
            clar_open=False,
            clar_visible=False,
            email_visible=True,   # show email UI only now
            email_status_text="📬 If you'd like me to email this report to you, enter your address and press **Send**, or press **Skip** at the end of the report.",           
            email_status_visible=True,
            run_top_visible=False,
            run_after_visible=False,
        )
        return

    # Any unexpected phase -> reset UI
    yield _build_outputs(
    status="Resetting...",
    report="",
    phase=PH_IDLE,
    mgr=mgr,
    clar_open=False, clar_visible=False,
    email_visible=False,
    run_top_visible=True, run_after_visible=False,
    email_status_text="", email_status_visible=False,  
    )
    return

async def on_send(report_md_text: str, to_email: str, phase: str):
    """
    Streams local email status above the email box and updates buttons.
    Outputs: [email_status_md, send_btn, skip_btn]
    """
    # 1) Guard: report not ready
    if phase != PH_WAIT_EMAIL:
        # generator: yield once, then return (no value)
        yield "⚠️ Report is not ready yet.", gr.update(), gr.update()
        return

    # 2) Guard: empty email
    to = (to_email or "").strip()
    if not to:
        yield "⚠️ Please enter a valid recipient email.", gr.update(), gr.update()
        return

    # 3) Immediate UI feedback while sending
    yield "⏳ Sending...", gr.update(value="Sending...", interactive=False), gr.update(interactive=False)

    # 4) Do the actual send
    prompt = (
        f"to_email: {to}\n"
        f"subject_hint: Deep Research Report\n"
        f"report_markdown:\n{report_md_text}"
    )
    try:
        result = await Runner.run(email_agent, prompt)
        status_obj = getattr(result, "tool_result", {"status": "success"})
        status = status_obj.get("status", "success")
        reason = status_obj.get("reason", "")

        if status == "success":
            # allow sending again, re-enable buttons
            yield "✅ Sent. Check your email :)", gr.update(value="Send again", interactive=True), gr.update(value="Skip", interactive=True)
            return

        if status == "skipped":
            yield "ℹ️ Email skipped.", gr.update(value="Send", interactive=True), gr.update(value="Skip", interactive=True)
            return

        yield f"⚠️ Email not sent: {reason or 'unknown error'}", gr.update(value="Send", interactive=True), gr.update(value="Skip", interactive=True)
        return

    except Exception as e:
        yield f"⚠️ {e}", gr.update(value="Send", interactive=True), gr.update(value="Skip", interactive=True)
        return

def on_skip():
    return "⏭️ Skipped. No email will be sent.", gr.update(value="Send", interactive=True), gr.update(value="Skipped", interactive=False)


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky", neutral_hue="slate"), title="Deep Research", css=STYLE_CSS) as ui:
    gr.Markdown("# 🔬 Deep Research")

    # Persistent states
    phase_state = gr.State(PH_IDLE)
    manager_state = gr.State()  # keep ResearchManager across clicks

    # Query + Run (TOP button, used only for phase 1)
    query_tb = gr.Textbox(
        label="What topic would you like to research?",
        placeholder="Topic to research...",
        lines=2,
        elem_id="query_box",          
    )
    run_btn_top = gr.Button("Run", variant="primary") # visible only in PH_IDLE

    # Clarifying questions + answers (hidden until 1st Run)
    # visible only in PH_WAIT_ANS
    with gr.Accordion("Clarifying Answers", open=False, visible=False) as clar_acc:
        q1_md = gr.Markdown(visible=False)
        a1 = gr.Textbox(label="Answer 1 (optional)")
        q2_md = gr.Markdown(visible=False)
        a2 = gr.Textbox(label="Answer 2 (optional)")
        q3_md = gr.Markdown(visible=False)
        a3 = gr.Textbox(label="Answer 3 (optional)")

    # Status 
    status_md = gr.Markdown(label="Status")
    # Run button AFTER the note (hidden initially; shown only in PH_WAIT_ANS)
    run_btn_after = gr.Button("Run", variant="primary", visible=False)
    # Report
    report_md = gr.Markdown(label="Report")

    # Email (hidden until report is ready)
    with gr.Group(visible=False, elem_id="email_row") as email_group:
        email_status_md = gr.Markdown(visible=False, elem_id="email_status")
        email_tb = gr.Textbox(label="Send report to email", placeholder="name@example.com")
        send_btn = gr.Button("Send")
        skip_btn = gr.Button("Skip")

    # Wire up TOP Run (phase 1)
    run_btn_top.click(
        fn=on_run,
        inputs=[query_tb, phase_state, a1, a2, a3, manager_state],
        outputs=[status_md, report_md, phase_state, manager_state, q1_md, q2_md, q3_md, clar_acc, email_group, run_btn_top, run_btn_after, email_status_md],
        queue=True,
    )
    query_tb.submit(
        fn=on_run,
        inputs=[query_tb, phase_state, a1, a2, a3, manager_state],
        outputs=[status_md, report_md, phase_state, manager_state, q1_md, q2_md, q3_md, clar_acc, email_group, run_btn_top, run_btn_after, email_status_md],
        queue=True,
    )

    # Wire up AFTER-NOTE Run (phase 2)
    run_btn_after.click(
        fn=on_run,
        inputs=[query_tb, phase_state, a1, a2, a3, manager_state],
        outputs=[status_md, report_md, phase_state, manager_state, q1_md, q2_md, q3_md, clar_acc, email_group, run_btn_top, run_btn_after, email_status_md],
        queue=True,
    )

    # Send / Skip
    send_btn.click(
        fn=on_send,
        inputs=[report_md, email_tb, phase_state],
        outputs=[email_status_md, send_btn, skip_btn],
        queue=True,
    )
    skip_btn.click(
        fn=on_skip,
        inputs=[],
        outputs=[email_status_md, send_btn, skip_btn],
        queue=False,
    )

# Expose Gradio demo for `gradio deploy`
demo = ui

if __name__ == "__main__":
    demo.launch(inbrowser=True)