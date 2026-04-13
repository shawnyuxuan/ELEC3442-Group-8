from dotenv import load_dotenv
import caldav
from caldav import get_davclient
from datetime import datetime, timedelta, time

load_dotenv()

class CalendarController:
    # 'Home' corresponds to the default calendar name in iCloud.
    def __init__(self, calendar_name="Home"):
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

    def update_event(self, event, new_title=None, new_start_time=None, new_end_time=None, new_description=None):
        if not event:
            raise Exception("No event provided to update.")
        
        # See https://github.com/python-caldav/caldav/blob/master/examples/basic_usage_examples.py#L266
        with event.edit_icalendar_instance() as cal:
            for comp in cal.subcomponents:
                if comp.name == "VEVENT":
                    if new_title:
                        comp["SUMMARY"] = new_title
                    if new_start_time:
                        comp["DTSTART"] = new_start_time
                    if new_end_time:
                        comp["DTEND"] = new_end_time
                    if new_description:
                        comp["DESCRIPTION"] = new_description
        event.save()
        print(f"Event updated successfully in calendar '{self.calendar_name}'.")
        
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
    calendar_event = CalendarController("Home")
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
        
