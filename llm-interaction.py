from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "gemini-2.5-flash"
client = genai.Client(vertexai=True)

def skill_1(*args, **kwargs):
    pass

sys_config = """
You are an adaptive schedule assistant for a university student.

Your job is to revise today's schedule based on:
1. A predicted sleep cluster label.
2. A short semantic description of the cluster.
3. A list of scheduled events and TODO tasks.
4. Basic constraints such as fixed events, deadlines, and task priority.

Rules:
- Do not diagnose medical conditions.
- Do not mention raw sleep records or private data.
- Do not invent tasks that are not in the schedule.
- Keep fixed events unchanged.
- Prefer practical, concise, and actionable suggestions.
- Reorder, shorten, defer, or group flexible tasks when needed.
- If the sleep state is weak or unstable, suggest protecting high-focus tasks and moving light tasks earlier.
- If the sleep state is strong, suggest keeping the original plan or expanding deep-work blocks.

Output format:
Return a JSON object with:
- "sleep_assessment": short summary of the predicted state
- "schedule_strategy": short explanation of how to adapt the day
- "recommended_changes": list of concrete changes
- "priority_order": reordered task list if needed
- "notes": brief caution or reminder

Be concise, structured, and practical.
"""

tools = [skill_1]

config = types.GenerateContentConfig(
    system_instruction=sys_config,
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
)

def build_prompt(cluster_id, cluster_label, cluster_meaning, schedule_text, constraints_text=""):
    return f"""
Predicted sleep cluster:
- cluster_id: {cluster_id}
- cluster_label: {cluster_label}
- cluster_meaning: {cluster_meaning}

Today's schedule:
{schedule_text}

Constraints:
{constraints_text}

Task:
Revise the schedule for today based on the predicted sleep state.
Keep fixed events unchanged.
Prioritize high-value tasks.
Suggest a practical adjusted schedule.

Return JSON only.
"""

prompt = build_prompt(
    cluster_id=0,
    cluster_label="RESTED_BALANCED",
    cluster_meaning="Sleep was balanced and restorative. Good energy for focused work.",
    schedule_text="""9:00 AM - 10:00 AM: Team Meeting
10:00 AM - 12:00 PM: Deep Work Session
12:00 PM - 1:00 PM: Lunch Break
1:00 PM - 3:00 PM: Project A
3:00 PM - 4:00 PM: Project B
4:00 PM - 5:00 PM: Wrap-up and Planning for Tomorrow""",
    constraints_text="""- Team Meeting is fixed
- Project A deadline is today
- Project B is flexible
- Wrap-up can be shortened"""
)

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=config,
)

print(response.text)
