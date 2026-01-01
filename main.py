# main.py
import os
from dotenv import load_dotenv

from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
)
from llama_index.readers.web import SimpleWebPageReader
from google import genai

# ---------------------------
# Load env vars
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

# ---------------------------
# Events
# ---------------------------
class UrlEvent(StartEvent):
    url: str

class PageContentEvent(Event):
    url: str
    content: str

class SummaryEvent(StopEvent):
    url: str
    summary: str

# ---------------------------
# Workflow
# ---------------------------
class UrlSummarizerWorkflow(Workflow):

    @step
    async def load_page(self, ev: UrlEvent) -> PageContentEvent:
        reader = SimpleWebPageReader()
        documents = reader.load_data([ev.url])

        # Combine all document text
        content = "\n".join(doc.text for doc in documents)

        # Keep within free-tier limits
        content = content[:6000]

        return PageContentEvent(url=ev.url, content=content)

    @step
    async def summarize(self, ev: PageContentEvent) -> SummaryEvent:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
        Summarize the following webpage content in concise bullet points.
        Focus on key ideas and avoid repetition.

        CONTENT:
        {ev.content}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return SummaryEvent(
            url=ev.url,
            summary=response.text.strip(),
        )

# ---------------------------
# Required entrypoint
# ---------------------------
workflow = UrlSummarizerWorkflow(timeout=60, verbose=True)
