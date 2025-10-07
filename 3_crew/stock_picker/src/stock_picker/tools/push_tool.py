from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests

# when set up a custom tool you have to first describe using a pedantic object,
# the schema of what will be passed in to your custom tool.
# And then you end up writing an _run method, which is going to take that schema as its parameters.
class PushNotificationInput(BaseModel):
    """A message to be sent to the user"""
    message: str = Field(..., description="The message to be sent to the user.")

class PushNotificationTool(BaseTool):
    """A tool to send a push notification to the user"""
    name: str = "Send a Push Notification"
    description: str = (
        "This tool is used to send a push notification to the user using Pushover API."
    )
    # args_schema is the schema of the input that will be passed in to the tool
    args_schema: Type[BaseModel] = PushNotificationInput

    # _run is the method that will be called when the tool is used and actually send the push notification.
    def _run(self, message: str) -> str: 
        """Send a push notification to the user"""
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"

        print(f"Push: {message}")
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
        return '{"notification": "ok"}'