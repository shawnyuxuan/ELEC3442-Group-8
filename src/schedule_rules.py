from src.llm_interaction import ScheduleItem


HIGH_INTENSITY_KEYWORDS = ("project", "assignment", "deadline", "report", "study")
MEDIUM_INTENSITY_KEYWORDS = ("meeting", "lecture", "class", "lab", "tutorial")
MOVABLE_KEYWORDS = ("project", "assignment", "task", "study", "review")


def infer_event_intensity(title: str, description: str) -> str:
    lowered = f"{title} {description}".lower()
    if any(keyword in lowered for keyword in HIGH_INTENSITY_KEYWORDS):
        return "high"
    if any(keyword in lowered for keyword in MEDIUM_INTENSITY_KEYWORDS):
        return "medium"
    return "low"


def infer_event_movable(title: str, description: str) -> bool:
    lowered = f"{title} {description}".lower()
    return any(keyword in lowered for keyword in MOVABLE_KEYWORDS)


def schedule_item_from_calendar_event(event) -> ScheduleItem:
    vevent = event.vobject_instance.vevent
    title = str(vevent.summary.value)
    start = vevent.dtstart.value.strftime("%H:%M")
    end = vevent.dtend.value.strftime("%H:%M")
    description = str(getattr(vevent, "description", None).value) if hasattr(vevent, "description") else ""

    return ScheduleItem(
        str(event.url),
        start,
        end,
        title,
        description,
        infer_event_intensity(title, description),
        infer_event_movable(title, description),
    )
