import gradio as gr
from sidekick import Sidekick

# Initializes a new Sidekick instance when the app loads.
# - Creates and awaits async setup to build the graph and load tools.
# - Returns the ready-to-use Sidekick object as session state.
# Ensures each user session has an isolated agent and resources.
async def setup():
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick

# Core message-processing callback invoked by the "Go" button.
# - Receives user input, success criteria, and chat history.
# - Calls Sidekick.run_superstep() to perform one graph iteration.
# - Returns the updated conversation transcript for display.
# Acts as the primary bridge between the frontend and backend reasoning flow.
async def process_message(sidekick, message, success_criteria, history):
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick

# Reset callback invoked when the "Reset" button is clicked.
# - Creates a new Sidekick instance.
# - Awaits async setup to initialize the graph and tools.
# - Returns an empty message, empty success criteria, and the new Sidekick instance.
# Ensures each user session has a fresh Sidekick agent.
async def reset():
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick

# Cleanup callback triggered when a Gradio session is closed.
# Ensures Playwright browser and other resources tied to this Sidekick instance
# are properly terminated to prevent leaks or lingering Chromium processes.
def free_resources(sidekick):
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")

# Defines the Gradio UI layout:
# - Chatbot component for dialogue visualization.
# - Textboxes for message input and success criteria.
# - Go / Reset buttons bound to callbacks.
# Uses Gradio's Blocks API with stateful components for multi-user sessions.
with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Sidekick Personal Co-Worker")
    sidekick = gr.State(delete_callback=free_resources)

    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success critiera?")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

   # Bind the callbacks to the UI elements (call the function, pass the arguments, return the outputs)
    ui.load(setup, [], [sidekick])
    message.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    success_criteria.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    go_button.click(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    reset_button.click(reset, [], [message, success_criteria, chatbot, sidekick])

# Entrypoint: launches the Gradio web server.
# Enables interactive Sidekick sessions directly from the browser.
# Use share=True for public demos or local debugging.
ui.launch(inbrowser=True)
