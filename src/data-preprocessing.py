import argparse
import datetime as dt
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter


TARGET_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_INPUT = "export.xml"
DEFAULT_OUTPUT_XML = "data_jayden.xml"
DEFAULT_OUTPUT_DATES = "dates_jayden.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sleep records for one data source from Apple Health export.xml."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to the Apple Health export XML. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--source",
        help="Exact sourceName match, for example: Jayden的Apple Watch",
    )
    parser.add_argument(
        "--contains",
        default="Jayden",
        help='Substring match for sourceName. Default: "Jayden"',
    )
    parser.add_argument(
        "--output-xml",
        default=DEFAULT_OUTPUT_XML,
        help=f"Filtered sleep XML output path. Default: {DEFAULT_OUTPUT_XML}",
    )
    parser.add_argument(
        "--output-dates",
        default=DEFAULT_OUTPUT_DATES,
        help=f"Session index JSON output path. Default: {DEFAULT_OUTPUT_DATES}",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available sleep sourceName values and exit.",
    )
    return parser.parse_args()


def parse_sleep_datetime(date_string: str) -> dt.datetime:
    return dt.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %z")


def in_same_session(previous_end: dt.datetime, current_start: dt.datetime) -> bool:
    return (current_start.timestamp() - previous_end.timestamp()) < 3600


def infer_weekday(session_start: dt.datetime) -> str:
    weekday_index = (
        (session_start.weekday() - 1) % 7
        if session_start.time() < dt.time(7, 0, 0)
        else session_start.weekday()
    )
    return WEEKDAYS[weekday_index]


def normalize_text(text: str) -> str:
    # Apple Health source names often mix non-breaking spaces and smart quotes.
    normalized = text.replace("\xa0", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def source_matches(source_name: str, exact_source: str | None, contains_text: str | None) -> bool:
    if not source_name:
        return False

    normalized_source = normalize_text(source_name)
    if exact_source:
        return normalized_source == normalize_text(exact_source)
    if contains_text:
        return normalize_text(contains_text) in normalized_source
    return True


def iterate_sleep_records(xml_path: str):
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if elem.tag != "Record":
            continue
        if elem.attrib.get("type") == TARGET_TYPE:
            yield dict(elem.attrib)
        elem.clear()


def list_sleep_sources(xml_path: str) -> Counter:
    sources = Counter()
    for attrib in iterate_sleep_records(xml_path):
        source_name = attrib.get("sourceName")
        if source_name:
            sources[source_name] += 1
    return sources


def collect_filtered_sleep_records(
    xml_path: str, exact_source: str | None, contains_text: str | None
) -> tuple[list[dict[str, str]], Counter]:
    records: list[dict[str, str]] = []
    sources = Counter()

    for attrib in iterate_sleep_records(xml_path):
        source_name = attrib.get("sourceName")
        if source_name:
            sources[source_name] += 1
        if source_matches(source_name or "", exact_source, contains_text):
            records.append(attrib)

    return records, sources


def build_dates_index(records: list[dict[str, str]]) -> list[list[object]]:
    if not records:
        return []

    start_dates = [parse_sleep_datetime(record["startDate"]) for record in records]
    end_dates = [parse_sleep_datetime(record["endDate"]) for record in records]

    dates: list[list[object]] = []
    previous_end = end_dates[0]
    current_indices = [0]
    current_weekday = infer_weekday(start_dates[0])

    for index, current_start in enumerate(start_dates[1:], start=1):
        current_end = end_dates[index]
        if in_same_session(previous_end, current_start):
            current_indices.append(index)
        else:
            dates.append([current_weekday, current_indices])
            current_indices = [index]
            current_weekday = infer_weekday(current_start)
        previous_end = current_end

    dates.append([current_weekday, current_indices])
    return dates


def save_raw_data(records: list[dict[str, str]], output_path: str) -> None:
    root = ET.Element("SleepData")
    for attrib in records:
        ET.SubElement(root, "Record", attrib=attrib)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    with open(output_path, "wb") as handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)


def save_date_table(dates: list[list[object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dates, handle, indent=4, ensure_ascii=False)


def format_source_counts(source_counts: Counter, limit: int = 10) -> str:
    if not source_counts:
        return "No sleep sources found."
    return "\n".join(f"{count:>6}  {source!r}" for source, count in source_counts.most_common(limit))


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        return 1

    if args.list_sources:
        print(format_source_counts(list_sleep_sources(args.input), limit=50))
        return 0

    records, source_counts = collect_filtered_sleep_records(args.input, args.source, args.contains)
    if not records:
        print("No matching sleep records found.", file=sys.stderr)
        print("Available sleep sources:", file=sys.stderr)
        print(format_source_counts(source_counts), file=sys.stderr)
        return 1

    dates = build_dates_index(records)
    save_raw_data(records, args.output_xml)
    save_date_table(dates, args.output_dates)

    matched_sources = Counter(record.get("sourceName") for record in records)
    print(f"Matched sleep records: {len(records)}")
    print(f"Matched sleep sessions: {len(dates)}")
    print("Matched sources:")
    print(format_source_counts(matched_sources, limit=20))
    print(f"Exported XML: {args.output_xml}")
    print(f"Exported dates: {args.output_dates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
