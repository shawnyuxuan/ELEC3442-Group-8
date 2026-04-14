from dataclasses import asdict, dataclass


@dataclass
class CalendarEventData:
    event_id: str | None
    title: str
    start: str
    end: str
    description: str
    intensity: str | None = None
    movable: bool | None = None

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            event_id=payload.get("event_id"),
            title=str(payload["title"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            description=str(payload["description"]),
            intensity=payload.get("intensity"),
            movable=payload.get("movable"),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class CalendarOperation:
    action: str
    target: CalendarEventData
    updated: CalendarEventData
    reason: str

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            action=str(payload["action"]),
            target=CalendarEventData.from_dict(payload["target"]),
            updated=CalendarEventData.from_dict(payload["updated"]),
            reason=str(payload["reason"]),
        )

    def to_dict(self):
        return {
            "action": self.action,
            "target": self.target.to_dict(),
            "updated": self.updated.to_dict(),
            "reason": self.reason,
        }


@dataclass
class ScheduleRecommendation:
    date: str
    daily_summary: str
    adjustment_principles: list[str]
    calendar_operations: list[CalendarOperation]
    optimized_schedule: list[CalendarEventData]

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            date=str(payload["date"]),
            daily_summary=str(payload["daily_summary"]),
            adjustment_principles=[str(item) for item in payload["adjustment_principles"]],
            calendar_operations=[
                CalendarOperation.from_dict(item) for item in payload["calendar_operations"]
            ],
            optimized_schedule=[
                CalendarEventData.from_dict(item) for item in payload["optimized_schedule"]
            ],
        )

    def to_dict(self):
        return {
            "date": self.date,
            "daily_summary": self.daily_summary,
            "adjustment_principles": list(self.adjustment_principles),
            "calendar_operations": [item.to_dict() for item in self.calendar_operations],
            "optimized_schedule": [item.to_dict() for item in self.optimized_schedule],
        }
