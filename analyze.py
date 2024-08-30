import asyncio
import json
import os
import requests
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime as dt
from pathlib import Path
from typing import Self
from yarl import URL

import discord
import gspread
import schedule
from dateutil.relativedelta import relativedelta as tdelta
from google.oauth2.service_account import Credentials
from gspread_formatting import format_cell_range, CellFormat, Color
from gspread.utils import ValueInputOption


from dotenv import load_dotenv

load_dotenv()


RAIDHELPER_TOKEN = os.environ["RAIDHELPER_TOKEN"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
DISCORD_SERVER_ID = int(os.environ["DISCORD_SERVER_ID"])
GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]

URL_BASE = URL("https://raid-helper.dev")
URL_EVENTS = URL_BASE / "api" / "v3" / "servers" / str(DISCORD_SERVER_ID) / "events"
URL_ATTENDANCE = (
    URL_BASE / "api" / "v2" / "servers" / str(DISCORD_SERVER_ID) / "attendance"
)

MONDAY = 0
TUESDAY = 1
WEDNESDAY = 2
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6
WEEKDAYS = {
    MONDAY: "monday",
    TUESDAY: "tuesday",
    WEDNESDAY: "wednesday",
    THURSDAY: "thursday",
    FRIDAY: "friday",
    SATURDAY: "saturday",
    SUNDAY: "sunday",
}


class Singleton:
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Member:
    id: int
    name: str
    joined_at: float = None

    @property
    def joined_formatted(self):
        return (
            str(dt.fromtimestamp(self.joined_at).date())
            if self.joined_at
            else "(not present)"
        )

    def __eq__(self, other):
        if isinstance(other, Member):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"{self.name}"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "joined_at": self.joined_at,
        }


@dataclass
class SignUp:
    id: int
    user_id: int
    name: str
    spec_name: str
    class_name: str
    entry_time: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "name": self.name,
            "specName": self.spec_name,
            "className": self.class_name,
            "entryTime": self.entry_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data["id"],
            user_id=int(data["userId"]),
            name=data["name"],
            spec_name=data["specName"],
            class_name=data["className"],
            entry_time=int(data["entryTime"]),
        )


@dataclass
class Event:
    id: str
    title: str
    description: str
    signup_count: str
    leader_id: str
    leader_name: str
    start_time: int
    end_time: int
    close_time: int
    last_updated: int
    sign_ups: list[SignUp] = field(default_factory=list)

    @property
    def weekday(self) -> int:
        return dt.fromtimestamp(self.start_time).weekday()

    @property
    def link(self) -> str:
        return URL_BASE / "event" / str(self.id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "signUpCount": self.signup_count,
            "leaderId": self.leader_id,
            "leaderName": self.leader_name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "closeTime": self.close_time,
            "lastUpdated": self.last_updated,
            "signUps": [s.to_dict() for s in self.sign_ups],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            signup_count=data["signUpCount"],
            leader_id=data["leaderId"],
            leader_name=data["leaderName"],
            start_time=data["startTime"],
            end_time=data["endTime"],
            close_time=data["closeTime"],
            last_updated=data["lastUpdated"],
            sign_ups=[SignUp.from_dict(_) for _ in data["signUps"]],
        )


@dataclass
class EventCollector:
    members: set[Member] = field(default_factory=set)
    member_event_participation: dict[int, list[str]] = field(
        default_factory=lambda: defaultdict(lambda: [])
    )
    sign_up_modes: set[str] = field(default_factory=set)
    events: list[Event] = field(default_factory=list)
    cur_ts: int = field(default_factory=lambda: dt.now().timestamp())
    calculated_attendance: dict = field(default_factory=dict)

    EXCLUDE_SIGNUP_MODES = ["Tentative", "Absence"]
    SIGNUP_MODES = {
        "Tentative": "T",
        "Absence": "A",
    }
    SIGN_DEFAULT = "✓"

    @property
    def cut_off_date(self):
        return dt.combine(date.today() - tdelta(months=3), dt.min.time())

    @property
    def future_events(self):
        return [e for e in self.events if e.end_time > self.cur_ts]

    @classmethod
    def signup_character(cls, mode):
        return cls.SIGNUP_MODES.get(mode, cls.SIGN_DEFAULT)

    def add(self, event_dict: dict) -> None:
        event = Event.from_dict(event_dict)
        self.events.append(event)

        for sign_up in event.sign_ups:
            member = Member(int(sign_up.user_id), sign_up.name)
            self.members.add(member)
            self.sign_up_modes.add(sign_up.class_name)
            if sign_up.class_name in self.EXCLUDE_SIGNUP_MODES:
                continue
            self.member_event_participation[str(sign_up.user_id)].append(event.id)

    def save(self, filename="data") -> None:
        export_to_file(
            {
                "ts": self.cur_ts,
                "members": list(map(lambda m: m.to_dict(), self.members)),
                "event_participation": self.member_event_participation,
                "sign_up_modes": list(self.sign_up_modes),
                "events": [e.to_dict() for e in self.events],
                "calculated_attendance": self.calculated_attendance,
            },
            filename,
        )

    @classmethod
    def load(cls, file="exports/data.json") -> Self:
        try:
            with open(Path.cwd() / file, "r") as f:
                data: dict = json.load(f)
        except FileNotFoundError:
            return None

        return EventCollector(
            members=set(
                Member(
                    int(m["id"]),
                    m["name"],
                    float(m["joined_at"]) if m["joined_at"] else None,
                )
                for m in data["members"]
            ),
            member_event_participation=data["event_participation"],
            sign_up_modes=set(data["sign_up_modes"]),
            events=[Event.from_dict(e) for e in data["events"]],
            calculated_attendance=data.get("calculated_attendance"),
            cur_ts=data.get("ts", dt.fromtimestamp(0)),
        )

    def fetch_date_joined(self) -> None:
        for disc_member in get_discord_member_data(m.id for m in self.members):
            member = next(m for m in self.members if m.id == disc_member.id)
            member.joined_at = disc_member.joined_at.timestamp()

    def calc_attendance(
        self,
        week_days: list[int] = None,
        previous_attendance: dict[str, str] = None,
    ) -> None:
        week_days = week_days or list(WEEKDAYS.keys())
        previous_attendance = previous_attendance or {}
        cut_off_ts = self.cut_off_date.timestamp()
        events = list(
            filter(
                lambda e: (
                    e.start_time >= cut_off_ts
                    and e.end_time < self.cur_ts
                    and e.weekday in week_days
                ),
                self.events,
            )
        )

        for m in self.members:
            # NOTE: exclude members that are no longer in this discord server:
            if m.joined_at is None:
                continue

            first_raid_ts = min(
                [
                    e.start_time
                    for e in self.events
                    if e.id in self.member_event_participation.get(str(m.id), [])
                ],
                default=self.cur_ts,
            )
            member_starting_point = max(m.joined_at, first_raid_ts)
            main_events_since = list(
                filter(lambda e: e.start_time >= member_starting_point, events)
            )

            events_participated = set(
                self.member_event_participation.get(str(m.id), [])
            ).intersection([e.id for e in main_events_since])

            self.calculated_attendance[m.name] = {
                "participation": f"{len(events_participated) / (len(main_events_since) or 1) * 100.0:.2f}%",
                "participated": len(events_participated),
                "total_raids_since_joining": len(main_events_since),
                "total_participation": f"{len(events_participated) / (len(events) or 1) * 100.0:.2f}%",
                "participation_prev": previous_attendance.get(m.name, {}).get("participation", "?"),
                "total_participation_prev": previous_attendance.get(m.name, {}).get("total_participation", "?"),
            }

    @classmethod
    def attendance_to_csv(cls, member_attendance: dict) -> list:
        csv = sorted(
            [
                [
                    name,
                    params["participation"],
                    params["total_participation"],
                    params["total_raids_since_joining"],
                    params["participated"],
                    params["participation_prev"],
                    params["total_participation_prev"],
                ]
                for name, params in member_attendance.items()
            ],
            key=lambda l: l[0],  # ], key=lambda l: -float(l[1][:-1]))
        )
        csv.insert(
            0,
            [
                "Name",
                "Attendance",
                "Overall attendance",
                "# of raids since joining",
                "# of raids participated",
                "Previous attendance",
                "Previous overall attendance",
            ],
        )
        return csv

    def members_to_csv(self):
        csv = [
            [m.name, str(m.id), m.joined_formatted]
            for m in sorted(list(self.members), key=lambda m: m.name)
        ]
        csv.insert(0, ["Name", "ID", "Joined discord"])
        return csv

    def raid_data_to_csv(self):
        members = sorted(self.members, key=lambda m: m.name)
        csv = []

        for e in sorted(self.events, key=lambda e: e.start_time, reverse=True):
            row = [
                rf'=HYPERLINK("{e.link}", "{dt.fromtimestamp(e.start_time).date()}")',
                e.title,
            ]
            signups_by_id = {s.user_id: s for s in e.sign_ups}
            row.extend(
                (self.signup_character(s.class_name) if s else "")
                for s in (signups_by_id.get(m.id) for m in members)
            )
            csv.append(row)
        csv.insert(0, ["Raid", "Title"] + [m.name for m in members])

        return csv

    def sign_up_modes_to_csv(self):
        csv = [[c, self.signup_character(c)] for c in sorted(list(self.sign_up_modes))]
        csv.insert(0, ["Classname", "Sign"])
        return csv


def get_discord_member_data(member_ids: list[int]) -> list[discord.Member]:
    members = []

    if not member_ids:
        return members

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(loop=asyncio.new_event_loop(), intents=intents)

    @client.event
    async def on_ready():
        try:
            guild = client.get_guild(DISCORD_SERVER_ID)
            if not guild:
                raise Exception(f"Discord server not found: '{DISCORD_SERVER_ID}'")

            for member_id in member_ids:
                if member := guild.get_member(member_id):
                    members.append(member)
        finally:
            await client.close()

    client.run(DISCORD_TOKEN)

    return members


def fetch_attendance() -> dict:
    return requests.get(
        URL_ATTENDANCE, headers={"Authorization": RAIDHELPER_TOKEN}
    ).json()


def fetch_events() -> EventCollector:
    data = EventCollector()
    initial_page = requests.get(
        URL_EVENTS,
        headers={"Authorization": RAIDHELPER_TOKEN, "Page": str(1)},
    ).json()

    for page in range(1, initial_page["pages"] + 1):
        resp = requests.get(
            URL_EVENTS,
            headers={
                "Authorization": RAIDHELPER_TOKEN,
                "Page": str(page),
                "IncludeSignUps": "True",
            },
        ).json()

        for e in resp["postedEvents"]:
            data.add(e)

    return data


def export_to_file(data: dict, filename: str, suffix: str = ".json") -> None:
    folder = Path.cwd() / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    filepath = (folder / filename).with_suffix(suffix)
    with open(filepath, "w+") as file:
        json.dump(data, file, indent=2)


class GoogleSheetExporter(Singleton):
    WORKSHEETS_MAX = 10

    def __init__(self) -> None:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        service_key = Path.cwd() / "google_auth" / "google_key.json"
        credentials = Credentials.from_service_account_file(service_key, scopes=scopes)
        self.gc = gspread.authorize(credentials)
        self.document = self.gc.open_by_key(GOOGLE_SHEET_ID)
        self.sheets = {}

    def sheet(self, index=0):
        if not isinstance(index, int) or not 0 <= index <= self.WORKSHEETS_MAX:
            raise Exception(f"Invalid worksheet index ({index})")
        if index in self.sheets:
            return self.sheets[index]
        try:
            sheet = self.document.get_worksheet(index)
            self.sheets[index] = sheet
            return sheet
        except gspread.exceptions.WorksheetNotFound:
            for i in range(len(self.document.worksheets()), index + 1):
                print("CREATING", i)
                self.sheets[i] = self.document.add_worksheet(
                    f"worksheet {i + 1}", 100, 20
                )
            return self.sheets[index]

    def paint_background(
        self,
        sheet_id: int,
        cell_range_a1: str,
        color: Color,
    ) -> None:
        format_cell_range(
            self.sheet(sheet_id), cell_range_a1, CellFormat(backgroundColor=color)
        )

    @staticmethod
    def int_to_a1(column: int) -> str:
        result = ""
        while column > 0:
            column -= 1
            result = chr(column % 26 + ord("A")) + result
            column //= 26
        return result


def job() -> None:
    print(f"\n\n{dt.now().ctime()}: Executing...")
    days = [TUESDAY, THURSDAY, FRIDAY, SATURDAY]
    data_last = EventCollector.load()
    attendance_last = data_last.calculated_attendance if data_last else None
    data = fetch_events()
    data.fetch_date_joined()
    data.calc_attendance(week_days=days, previous_attendance=attendance_last)
    data.save()

    exporter = GoogleSheetExporter()
    exporter.document.update_title(f"Raid data (updated @ {dt.now().ctime()})")

    # tab 1:
    exporter.sheet(0).clear()
    exporter.sheet(0).update_title(f"Participation")
    exporter.sheet(0).insert_rows(
        EventCollector.attendance_to_csv(data.calculated_attendance)
    )
    exporter.sheet(0).freeze(rows=1)

    # tab 2:
    exporter.sheet(1).clear()
    exporter.sheet(1).update_title("All raids")
    exporter.sheet(1).insert_rows(
        data.raid_data_to_csv(),
        value_input_option=ValueInputOption.user_entered,
    )
    exporter.sheet(1).freeze(rows=1, cols=2)
    exporter.paint_background(
        1,
        f"A{2}:{exporter.int_to_a1(len(data.members) + 2)}{len(data.future_events) + 1}",
        Color(0.7, 0.7, 0.7),
    )

    # tab 3:
    exporter.sheet(2).clear()
    exporter.sheet(2).update_title("Members")
    exporter.sheet(2).insert_rows(data.members_to_csv())
    exporter.sheet(2).freeze(rows=1)

    # tab 4:
    exporter.sheet(3).clear()
    exporter.sheet(3).update_title("Sign-up types")
    exporter.sheet(3).insert_rows(data.sign_up_modes_to_csv())
    exporter.sheet(3).freeze(rows=1)

    print(f"{dt.now().ctime()}: Done.\n---")


def main() -> None:
    if not all(
        map(bool, (RAIDHELPER_TOKEN, DISCORD_TOKEN, DISCORD_SERVER_ID, GOOGLE_SHEET_ID))
    ):
        raise Exception("Not all env vars are set!")

    print("Running...")
    schedule.every().wednesday.at("03:00", "Europe/Amsterdam").do(job)
    schedule.every().friday.at("03:00", "Europe/Amsterdam").do(job)
    schedule.every().saturday.at("03:00", "Europe/Amsterdam").do(job)
    schedule.every().sunday.at("03:00", "Europe/Amsterdam").do(job)
    while True:
        schedule.run_pending()
        time.sleep(600)


if __name__ == "__main__":
    main()
    # job()
