from pydantic import BaseModel
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import requests


def to_markdown(data, indent=0):
    markdown = ""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    if isinstance(data, dict):
        for key, value in data.items():
            markdown += f"{'#' * (indent + 2)} {key.upper()}\n"
            if isinstance(value, (dict, list, BaseModel)):
                markdown += to_markdown(value, indent + 1)
            else:
                markdown += f"{value}\n\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list, BaseModel)):
                markdown += to_markdown(item, indent)
            else:
                markdown += f"- {item}\n"
        markdown += "\n"
    else:
        markdown += f"{data}\n\n"
    return markdown


def send_email(subject: str, body: str, to_email: str):
    # Specify the path to the .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

    # Load environment variables from the specified .env file
    load_dotenv(dotenv_path)

    return requests.post(
        "https://api.mailgun.net/v3/sandboxa87742fadda649ab9e3350437d05cf58.mailgun.org/messages",
        auth=("api", os.getenv("MAILGUN_API_KEY")),
        data={"from": "AI analysis <mailgun@sandboxa87742fadda649ab9e3350437d05cf58.mailgun.org>",
            "to": to_email,
            "subject": subject,
            "text": body})