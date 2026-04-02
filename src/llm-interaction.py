# A skeleton only. Fill in the system instruction and prompt, and add skills as needed.
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "gemini-2.5-flash"
client = genai.Client(vertexai=True) # Google AI studio restricted in HK, so we need to use Vertex AI.

# Skills
def skill_1(*args, **kwargs):
    """skill example"""
    pass

# System instruction
sys_config = """

"""
tools = [skill_1]

config = types.GenerateContentConfig(
    system_instruction=sys_config,
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
)

prompt = """

"""

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=config,
)

print(response.text)