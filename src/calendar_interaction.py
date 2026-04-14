from dotenv import load_dotenv
import caldav
import os
from caldav import get_davclient
from datetime import date, datetime, timedelta, time
from src.llm_contract import CalendarEventData, CalendarOperation, ScheduleRecommendation

load_dotenv()

DEFAULT_CALENDAR_NAME = os.getenv("CALENDAR_NAME", "HKU Schedule")

class CalendarController:
    def __init__(self, calendar_name=DEFAULT_CALENDAR_NAME):
        self.client = self.create_client()
        self.calendar_name = calendar_name
        self.calendar = self.get_calendar(self.calendar_name)

    def create_client(self):
        client = get_davclient()
        return client
    
    def get_calendar(self, calendar_name):
        principal = self.client.principal()
        calendars = principal.get_calendars()
        
        for cal in calendars:
            if cal.get_display_name() == calendar_name:
                return cal
            
        raise Exception(f"Calendar '{calendar_name}' not found.")

    def create_event(self, event_title, start_time, end_time, description):
        new_event = self.calendar.add_event(
            dtstart=start_time,
            dtend=end_time,
            summary=event_title,
            description=description
        )
        print(f"Event '{event_title}' created successfully in calendar '{self.calendar_name}'.\n")
        return new_event

    def _rebuild_datetime_like(self, original_value, hhmm_text):
        parsed_time = datetime.strptime(hhmm_text, "%H:%M").time()
        if isinstance(original_value, datetime):
            return datetime.combine(original_value.date(), parsed_time, original_value.tzinfo)
        if isinstance(original_value, date):
            return datetime.combine(original_value, parsed_time)
        raise Exception(f"Unsupported calendar time value: {type(original_value).__name__}")

    def _build_modified_description_from_text(self, original_description: str, operation: CalendarOperation):
        updated_description = operation.updated.description.strip()
        if updated_description:
            return updated_description

        reason = operation.reason.strip()
        if not reason:
            return original_description

        note = f"[Sleep-aware adjustment] {reason}"
        if original_description:
            return f"{original_description}\n\n{note}"
        return note

    def _build_modified_description(self, event, operation: CalendarOperation):
        vevent = event.vobject_instance.vevent
        original_description = str(getattr(vevent, "description", None).value) if hasattr(vevent, "description") else ""
        return self._build_modified_description_from_text(original_description, operation)

    def update_event(self, event, new_title=None, new_start_time=None, new_end_time=None, new_description=None):
        if not event:
            raise Exception("No event provided to update.")
        
        # See https://github.com/python-caldav/caldav/blob/master/examples/basic_usage_examples.py#L266
        with event.edit_icalendar_component() as comp:
            if new_title:
                comp["SUMMARY"] = new_title
            if new_start_time:
                comp.pop("DTSTART", None)
                comp.add("DTSTART", new_start_time)
            if new_end_time:
                comp.pop("DTEND", None)
                comp.add("DTEND", new_end_time)
            if new_description is not None:
                comp.pop("DESCRIPTION", None)
                if new_description:
                    comp.add("DESCRIPTION", new_description)
        event.save()
        print(f"Event updated successfully in calendar '{self.calendar_name}'.")

    def _extract_event_fields(self, event):
        vevent = event.vobject_instance.vevent
        summary = str(vevent.summary.value)
        dtstart = vevent.dtstart.value
        dtend = vevent.dtend.value
        description = str(getattr(vevent, "description", None).value) if hasattr(vevent, "description") else ""
        return {
            "title": summary,
            "start": dtstart.strftime("%H:%M"),
            "end": dtend.strftime("%H:%M"),
            "description": description,
        }

    def _event_data_to_fields(self, event_data: CalendarEventData):
        return {
            "title": event_data.title,
            "start": event_data.start,
            "end": event_data.end,
            "description": event_data.description,
        }

    def _build_expected_updated_event(self, operation: CalendarOperation) -> CalendarEventData:
        if operation.action == "restore":
            return operation.updated

        if operation.action in ("move", "update_description"):
            return CalendarEventData(
                event_id=operation.updated.event_id,
                title=operation.updated.title,
                start=operation.updated.start,
                end=operation.updated.end,
                description=self._build_modified_description_from_text(
                    operation.target.description,
                    operation,
                ),
                intensity=operation.updated.intensity,
                movable=operation.updated.movable,
            )

        raise Exception(f"Unsupported calendar operation: {operation.action}")

    def _build_inverse_operation(self, operation: CalendarOperation) -> CalendarOperation:
        expected_updated = self._build_expected_updated_event(operation)
        return CalendarOperation(
            action="restore",
            target=expected_updated,
            updated=operation.target,
            reason=f"Rollback of: {operation.reason}",
        )

    def find_event_by_fields(self, events, expected_fields):
        for event in events:
            extracted = self._extract_event_fields(event)
            if (
                extracted["title"] == expected_fields["title"]
                and extracted["start"] == expected_fields["start"]
                and extracted["end"] == expected_fields["end"]
                and extracted["description"] == expected_fields["description"]
            ):
                return event
        return None

    def find_event_by_target(self, events, target):
        return self.find_event_by_fields(events, self._event_data_to_fields(target))

    def resolve_operation_from_events(self, events, operation: CalendarOperation):
        event = self.find_event_by_target(events, operation.target)
        if event is not None:
            return {
                "status": "target",
                "event": event,
                "inverse_operation": self._build_inverse_operation(operation),
            }

        already_updated = self.find_event_by_target(events, self._build_expected_updated_event(operation))
        if already_updated is not None:
            return {
                "status": "updated",
                "event": already_updated,
                "inverse_operation": None,
            }

        return {
            "status": "missing",
            "event": None,
            "inverse_operation": None,
        }

    def preflight_operations(self, date_start, operations: list[CalendarOperation]):
        events = None
        preflight = []
        for operation in operations:
            if operation.action == "no_update":
                preflight.append(
                    {
                        "status": "no_update",
                        "event": None,
                        "inverse_operation": None,
                    }
                )
                continue
            if events is None:
                events = self.fetch_events(date_start)
            resolved = self.resolve_operation_from_events(events, operation)
            if resolved["status"] == "missing":
                raise Exception(
                    f"Could not find event matching target '{operation.target.title}' "
                    f"from {operation.target.start} to {operation.target.end}."
                )
            preflight.append(resolved)
        return preflight

    def apply_calendar_operation(self, date_start, operation: CalendarOperation):
        if operation.action == "no_update":
            print(
                f"No calendar update needed for calendar '{self.calendar_name}'. "
                f"Reason: {operation.reason}"
            )
            return

        events = self.fetch_events(date_start)
        resolved = self.resolve_operation_from_events(events, operation)
        event = resolved["event"]
        if resolved["status"] == "updated":
            print(
                f"Calendar operation for '{operation.target.title}' already appears to be applied in "
                f"calendar '{self.calendar_name}'."
            )
            return
        if resolved["status"] == "missing" or event is None:
            raise Exception(
                f"Could not find event matching target '{operation.target.title}' "
                f"from {operation.target.start} to {operation.target.end}."
            )
        if operation.action in ("move", "restore"):
            vevent = event.vobject_instance.vevent
            self.update_event(
                event=event,
                new_title=operation.updated.title,
                new_start_time=self._rebuild_datetime_like(vevent.dtstart.value, operation.updated.start),
                new_end_time=self._rebuild_datetime_like(vevent.dtend.value, operation.updated.end),
                new_description=(
                    operation.updated.description
                    if operation.action == "restore"
                    else self._build_modified_description(event, operation)
                ),
            )
            return

        if operation.action == "update_description":
            self.update_event(
                event=event,
                new_description=self._build_modified_description(event, operation),
            )
            return

        raise Exception(f"Unsupported calendar operation: {operation.action}")

    def apply_schedule_recommendation(self, recommendation: ScheduleRecommendation):
        date_start = datetime.combine(
            datetime.strptime(recommendation.date, "%Y-%m-%d").date(),
            time.min,
        )
        for operation in recommendation.calendar_operations:
            self.apply_calendar_operation(date_start, operation)
        
    def fetch_events(self, current_date):
        events = self.calendar.search(
            start=current_date,
            end=current_date+timedelta(days=1),
            event=True
        )
        
        print(f"Events on {current_date.date()} in calendar '{self.calendar_name}':")
        for event in events:
            # Old version of caldav library uses event.vobject_instance, which requires an additional library vobject. 
            # The behaviour does not change after all. For reference only.
            print(f"- {event.vobject_instance.vevent.summary.value} from {event.vobject_instance.vevent.dtstart.value} to {event.vobject_instance.vevent.dtend.value}")
        print()
        
        return events

# For test only.
# In production, this file should only export the CalendarController class and will not be executed directly.
if __name__ == "__main__":
    calendar_event = CalendarController(DEFAULT_CALENDAR_NAME)
    # Start of the date at 00:00:00
    date_start_time = datetime.combine(datetime.now().date(), time.min)
    
    start_time = datetime.now()
    try:
        create_event = calendar_event.create_event(
            event_title="Test Event",
            start_time=start_time + timedelta(hours=1),
            end_time=start_time + timedelta(hours=2),
            description="This is a test event created by the script."
        )
        events = calendar_event.fetch_events(date_start_time)
        calendar_event.update_event(
            event=events[0] if events else None,
            new_title="Updated Test Event",
            new_description="This is an updated description for the test event."
        )
        
    except Exception as e:
        print(f"Error: {e}")
        
