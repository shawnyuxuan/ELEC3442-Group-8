# The agent file creates a pipeline that 
# 1. listens for daily sleep reports and processes them using the daily-analysis module. 
# 2. uses the processed data to interact with LLM for insights and recommendations.
# 3. analyze the LLM feedback and make modifications to the user's calendar for better sleep hygiene.

from src.daily_analysis import app
from src.calendar_interaction import CalendarController
from src.llm_interaction import *
from src.sensehat_behavior import SenseHatController

import datetime
import os

