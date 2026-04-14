from src.llm_interaction import ScheduleItem


BLACKLIST_KEYWORDS = ("elec", "comp", "meeting", "interview")


def infer_event_intensity(title: str, description: str) -> str:
    return "medium"


def infer_event_movable(title: str, description: str) -> bool:
    lowered = f"{title} {description}".lower()
    return not any(keyword in lowered for keyword in BLACKLIST_KEYWORDS)


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
