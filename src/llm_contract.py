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
        try:
            required_fields = ["event_id", "title", "start", "end", "description"]
            for field in required_fields:
                if field not in payload:
                    raise ValueError(f"Missing required field: '{field}'")
            
            return cls(
                event_id=payload.get("event_id"),
                title=str(payload["title"]),
                start=str(payload["start"]),
                end=str(payload["end"]),
                description=str(payload["description"]),
                intensity=payload.get("intensity"),
                movable=payload.get("movable"),
            )
        except Exception as e:
            raise ValueError(f"Error parsing CalendarEventData: {e}") from e

    def to_dict(self):
        return asdict(self)


@dataclass
class CalendarOperation:
    action: str
    target: CalendarEventData | None
    updated: CalendarEventData | None
    reason: str

    @classmethod
    def from_dict(cls, payload: dict):
        try:
            target_payload = payload.get("target")
            updated_payload = payload.get("updated")
            
            # Validate required fields
            if "action" not in payload:
                raise ValueError("Missing 'action' field in calendar_operations")
            if "reason" not in payload:
                raise ValueError("Missing 'reason' field in calendar_operations")
            
            # Create CalendarEventData objects
            target = None
            updated = None
            
            if target_payload is not None:
                if not isinstance(target_payload, dict):
                    raise ValueError(f"'target' must be a dict or null, got {type(target_payload).__name__}")
                target = CalendarEventData.from_dict(target_payload)
            
            if updated_payload is not None:
                if not isinstance(updated_payload, dict):
                    raise ValueError(f"'updated' must be a dict or null, got {type(updated_payload).__name__}")
                updated = CalendarEventData.from_dict(updated_payload)
            
            return cls(
                action=str(payload["action"]),
                target=target,
                updated=updated,
                reason=str(payload["reason"]),
            )
        except Exception as e:
            raise ValueError(f"Error parsing calendar_operations: {e}") from e

    def to_dict(self):
        return {
            "action": self.action,
            "target": None if self.target is None else self.target.to_dict(),
            "updated": None if self.updated is None else self.updated.to_dict(),
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
