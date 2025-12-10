from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants & types
# ---------------------------------------------------------------------------

# You must honor this minimum layover between flights when searching.
MIN_LAYOVER_MINUTES: int = 60

Cabin = Literal["economy", "business", "first"]


@dataclass(frozen=True)
class Flight:
    """
    One scheduled flight (single day, same-day arrival).

    Times are stored as minutes since midnight (0–1439).
    """

    # TODO: verify you understand these fields and update docstrings as needed.
    origin: str
    dest: str
    flight_number: str
    depart: int  # minutes since midnight
    arrive: int  # minutes since midnight
    economy: int
    business: int
    first: int

    def price_for(self, cabin: Cabin) -> int:
        """
        Return the price for this flight in the given cabin.

        Hint:
        - Map cabin names to the numeric fields defined above.
        """
        if cabin == "economy":
            return self.economy
        if cabin == "business":
            return self.business
        if cabin == "first":
            return self.first
        raise ValueError(f"unknown cabin: {cabin}")


@dataclass
class Itinerary:
    """
    A sequence of one or more flights representing a full journey.

    You should assume:
    - flights are in chronological order.
    - the destination of each flight matches the origin of the next.
    """

    flights: List[Flight]

    def is_empty(self) -> bool:
        return not self.flights

    @property
    def origin(self) -> Optional[str]:
        # TODO: return the origin airport code of the first flight, or None.
        return None if not self.flights else self.flights[0].origin

    @property
    def dest(self) -> Optional[str]:
        # TODO: return the destination airport code of the last flight, or None.
        return None if not self.flights else self.flights[-1].dest

    @property
    def depart_time(self) -> Optional[int]:
        # TODO: return the departure time (minutes) of the first flight, or None.
        return None if not self.flights else self.flights[0].depart

    @property
    def arrive_time(self) -> Optional[int]:
        # TODO: return the arrival time (minutes) of the last flight, or None.
        return None if not self.flights else self.flights[-1].arrive

    def total_price(self, cabin: Cabin) -> int:
        """
        Sum the price of all flights in this itinerary for the given cabin.
        """
        return sum(f.price_for(cabin) for f in self.flights)

    def num_stops(self) -> int:
        """
        Number of stops = flights - 1.

        Example:
        - 1 flight: 0 stops (direct).
        - 3 flights: 2 stops.
        """
        return max(0, len(self.flights) - 1)


# Graph type: adjacency list mapping airport code -> list of outgoing flights.
Graph = Dict[str, List[Flight]]

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_time(hhmm: str) -> int:
    """
    Parse a time string 'HH:MM' (24-hour) into minutes since midnight.

    Examples:
        '00:00' -> 0
        '08:30' -> 510
        '23:59' -> 23*60 + 59
    """
    if not isinstance(hhmm, str):
        raise ValueError("time must be a string in HH:MM format")
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid time format: {hhmm}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        raise ValueError(f"invalid time numbers: {hhmm}")
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"time out of range: {hhmm}")
    return hour * 60 + minute


def format_time(minutes: int) -> str:
    """
    Convert minutes since midnight to 'HH:MM' (24-hour).
    """
    if minutes is None:
        return "N/A"
    if not isinstance(minutes, int):
        raise ValueError("minutes must be int")
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


# ---------------------------------------------------------------------------
# Loading flights from files
# ---------------------------------------------------------------------------


def parse_flight_line_txt(line: str) -> Optional[Flight]:
    """
    Parse a single space-separated flight line.

    Format:
        ORIGIN DEST FLIGHT_NUMBER DEPART ARRIVE ECONOMY BUSINESS FIRST

    Behavior:
    - Return a Flight if the line contains data.
    - Return None for:
        * blank lines
        * comment lines starting with '#'
    - Raise ValueError for malformed data lines.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) != 8:
        raise ValueError(f"malformed flight line (expected 8 fields): {line!r}")
    origin, dest, flight_number, depart_s, arrive_s, econ_s, bus_s, first_s = parts
    depart = parse_time(depart_s)
    arrive = parse_time(arrive_s)
    econ = int(econ_s)
    bus = int(bus_s)
    first = int(first_s)
    if arrive <= depart:
        raise ValueError(f"arrival must be after depart (same-day): {line!r}")
    return Flight(origin=origin, dest=dest, flight_number=flight_number, depart=depart, arrive=arrive, economy=econ, business=bus, first=first)


def load_flights_txt(path: str) -> List[Flight]:
    """
    Load flights from a plain text schedule file.
    """
    flights: List[Flight] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                f = parse_flight_line_txt(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}")
            if f is not None:
                flights.append(f)
    return flights


def load_flights_csv(path: str) -> List[Flight]:
    """
    Load flights from a CSV file with header:

        origin,dest,flight_number,depart,arrive,economy,business,first
    """
    flights: List[Flight] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = ["origin", "dest", "flight_number", "depart", "arrive", "economy", "business", "first"]
        for c in required:
            if c not in reader.fieldnames:  # type: ignore[arg-type]
                raise ValueError(f"CSV missing required column: {c}")
        for lineno, row in enumerate(reader, start=2):
            try:
                origin = row["origin"].strip()
                dest = row["dest"].strip()
                flight_number = row["flight_number"].strip()
                depart = parse_time(row["depart"].strip())
                arrive = parse_time(row["arrive"].strip())
                econ = int(row["economy"].strip())
                bus = int(row["business"].strip())
                first = int(row["first"].strip())
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: invalid CSV row: {e}")
            if arrive <= depart:
                raise ValueError(f"{path}:{lineno}: arrival must be after depart (same-day)")
            flights.append(Flight(origin=origin, dest=dest, flight_number=flight_number, depart=depart, arrive=arrive, economy=econ, business=bus, first=first))
    return flights


def load_flights(path: str) -> List[Flight]:
    """
    Wrapper that chooses TXT or CSV loader based on file extension.
    """
    suf = Path(path).suffix.lower()
    if suf == ".csv":
        return load_flights_csv(path)
    return load_flights_txt(path)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(flights: Iterable[Flight]) -> Graph:
    """
    Build an adjacency-list graph from a collection of flights.
    """
    g: Graph = {}
    for f in flights:
        g.setdefault(f.origin, []).append(f)
        # ensure destination node appears (optional) so airport presence checks work
        g.setdefault(f.dest, g.get(f.dest, []))
    # Optionally sort outgoing flights by departure time for slight efficiency
    for origin in list(g.keys()):
        g[origin].sort(key=lambda fl: fl.depart)
    return g


# ---------------------------------------------------------------------------
# Search functions (earliest arrival / cheapest)
# ---------------------------------------------------------------------------


def find_earliest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
) -> Optional[Itinerary]:
    """
    Dijkstra-like search where distance = earliest arrival time at an airport.
    """
    import math

    if start not in graph or dest not in graph:
        # If either airport is unknown in the graph, no route exists
        return None

    # dist[airport] = earliest time we can be at airport
    dist: Dict[str, int] = {node: math.inf for node in graph}
    prev: Dict[str, Optional[Flight]] = {node: None for node in graph}

    dist[start] = earliest_departure
    # priority queue of (time, airport)
    heap: List[Tuple[int, str]] = [(earliest_departure, start)]

    while heap:
        time_at_u, u = heapq.heappop(heap)
        if time_at_u != dist[u]:
            continue
        if u == dest:
            break
        # compute minimum allowable departure time from u
        if u == start:
            min_dep = time_at_u
        else:
            min_dep = time_at_u + MIN_LAYOVER_MINUTES
        for f in graph.get(u, []):
            if f.depart < min_dep:
                continue
            v = f.dest
            arrival = f.arrive
            if arrival < dist[v]:
                dist[v] = arrival
                prev[v] = f
                heapq.heappush(heap, (arrival, v))

    if dist[dest] == float("inf"):
        return None

    # Reconstruct itinerary: follow prev from dest back to start
    flights_rev: List[Flight] = []
    cur = dest
    while prev[cur] is not None:
        f = prev[cur]
        flights_rev.append(f)
        cur = f.origin
    flights = list(reversed(flights_rev))
    return Itinerary(flights=flights) if flights else None


def find_cheapest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
    cabin: Cabin,
) -> Optional[Itinerary]:
    """
    Dijkstra over (airport, arrival_time) states keyed by total cost.

    We maintain dominance lists per airport to prune dominated states:
    A state (t_arrive, cost) is dominated if there exists another state with
    arrive_time <= t_arrive and cost <= cost.
    """
    if start not in graph or dest not in graph:
        return None

    # Priority queue: (cost_so_far, airport, arrival_time)
    heap: List[Tuple[int, str, int]] = [(0, start, earliest_departure)]

    # For reconstructing path: key a state by (airport, arrival_time)
    prev: Dict[Tuple[str, int], Tuple[Optional[Tuple[str, int]], Flight]] = {}

    # For each airport keep a list of non-dominated (arrival_time, cost) pairs
    non_dominated: Dict[str, List[Tuple[int, int]]] = {node: [] for node in graph}

    while heap:
        cost, u, arrival_time = heapq.heappop(heap)
        # Check if this state is dominated by an already-seen better one
        dominated = False
        for (t_exist, c_exist) in non_dominated[u]:
            if t_exist <= arrival_time and c_exist <= cost:
                dominated = True
                break
        if dominated:
            continue

        # Add this state to non_dominated and remove any it dominates
        new_list: List[Tuple[int, int]] = []
        for (t_exist, c_exist) in non_dominated[u]:
            # keep existing if it's not dominated by new state
            if not (arrival_time <= t_exist and cost <= c_exist):
                new_list.append((t_exist, c_exist))
        new_list.append((arrival_time, cost))
        non_dominated[u] = new_list

        # If we reached destination, this is the least-cost arrival because
        # heap pops states by increasing cost
        if u == dest:
            # reconstruct path using prev: the state key is (u, arrival_time)
            flights_rev: List[Flight] = []
            cur_state = (u, arrival_time)
            while cur_state in prev:
                prev_state, flight_used = prev[cur_state]
                flights_rev.append(flight_used)
                if prev_state is None:
                    break
                cur_state = prev_state
            flights = list(reversed(flights_rev))
            return Itinerary(flights=flights) if flights else None

        # compute minimum allowable departure time from u
        if u == start:
            min_dep = arrival_time
        else:
            min_dep = arrival_time + MIN_LAYOVER_MINUTES

        for f in graph.get(u, []):
            if f.depart < min_dep:
                continue
            v = f.dest
            new_cost = cost + f.price_for(cabin)
            new_arrival = f.arrive
            # state is (v, new_arrival)
            state = (v, new_arrival)
            # quick dominance check against non_dominated[v]
            skip = False
            for (t_exist, c_exist) in non_dominated[v]:
                if t_exist <= new_arrival and c_exist <= new_cost:
                    skip = True
                    break
            if skip:
                continue
            # record predecessor
            prev[state] = ((u, arrival_time), f)
            heapq.heappush(heap, (new_cost, v, new_arrival))

    return None


# ---------------------------------------------------------------------------
# Formatting the comparison table
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[Cabin]  # e.g. None for earliest-arrival if you want
    itinerary: Optional[Itinerary]
    note: str = ""  # e.g. "(no valid itinerary)"


def _format_duration(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h}h{m:02d}m"


def format_comparison_table(
    origin: str,
    dest: str,
    earliest_departure: int,
    rows: List[ComparisonRow],
) -> str:
    """
    Format a text table comparing several itineraries.
    """
    header = f"Comparison for {origin} → {dest} (earliest departure {format_time(earliest_departure)}, layover ≥ {MIN_LAYOVER_MINUTES} min)"
    cols = [
        "Mode",
        "Cabin",
        "Dep",
        "Arr",
        "Duration",
        "Stops",
        "Total Price",
        "Note",
    ]
    # column widths
    widths = [22, 8, 6, 6, 10, 6, 11, 20]
    def pad(s: str, w: int) -> str:
        return s + " " * (w - len(s)) if len(s) < w else s[:w]

    lines: List[str] = [header, ""]
    # header row
    hrow = "  ".join(pad(c, w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    lines.append(hrow)
    lines.append(sep)

    for r in rows:
        if r.itinerary is None:
            dep = arr = duration = stops = price = "N/A"
            note = r.note or "(no valid itinerary)"
        else:
            it = r.itinerary
            dep_min = it.depart_time
            arr_min = it.arrive_time
            dep = format_time(dep_min) if dep_min is not None else "N/A"
            arr = format_time(arr_min) if arr_min is not None else "N/A"
            duration = _format_duration(arr_min - dep_min) if (dep_min is not None and arr_min is not None) else "N/A"
            stops = str(it.num_stops())
            price = str(it.total_price(r.cabin)) if r.cabin is not None else "N/A"
            note = r.note
        cabin = (r.cabin.capitalize() if r.cabin is not None else "-")
        row_elems = [r.mode, cabin, dep, arr, duration, stops, price, note]
        row = "  ".join(pad(str(e), w) for e, w in zip(row_elems, widths))
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> None:
    """
    Handle the 'compare' subcommand.
    """
    try:
        earliest_departure = parse_time(args.departure_time)
    except ValueError as e:
        print(f"Invalid departure_time: {e}")
        return

    try:
        flights = load_flights(args.flight_file)
    except Exception as e:
        print(f"Error loading flights: {e}")
        return

    graph = build_graph(flights)

    # verify airports exist somewhere in the data (either as origin or dest)
    airports = set(graph.keys())
    if args.origin not in airports:
        print(f"Unknown origin airport: {args.origin}")
        return
    if args.dest not in airports:
        print(f"Unknown destination airport: {args.dest}")
        return

    rows: List[ComparisonRow] = []

    # Earliest arrival (no cabin preference). We'll display Economy in cabin col for readability.
    ea = find_earliest_itinerary(graph, args.origin, args.dest, earliest_departure)
    rows.append(ComparisonRow(mode="Earliest arrival", cabin="economy", itinerary=ea, note="" if ea else "(no valid itinerary)"))

    # Cheapest per cabin
    for cabin in ("economy", "business", "first"):
        it = find_cheapest_itinerary(graph, args.origin, args.dest, earliest_departure, cabin)  # type: ignore[arg-type]
        mode_name = f"Cheapest ({cabin.capitalize()})"
        rows.append(ComparisonRow(mode=mode_name, cabin=cabin, itinerary=it, note="" if it else "(no valid itinerary)"))

    table = format_comparison_table(args.origin, args.dest, earliest_departure, rows)
    print(table)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser with a 'compare' subcommand.

    You generally do NOT need to change this unless you add features.
    """
    parser = argparse.ArgumentParser(
        description="FlyWise — Flight Route & Fare Comparator (Project 3)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare itineraries for a route (earliest arrival, cheapest per cabin).",
    )
    compare_parser.add_argument(
        "flight_file",
        help="Path to the flight schedule file (.txt or .csv).",
    )
    compare_parser.add_argument(
        "origin",
        help="Origin airport code (e.g., ICN).",
    )
    compare_parser.add_argument(
        "dest",
        help="Destination airport code (e.g., SFO).",
    )
    compare_parser.add_argument(
        "departure_time",
        help="Earliest allowed departure time (HH:MM, 24-hour).",
    )
    compare_parser.set_defaults(func=run_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the CLI.

    Example usage:
        python flight_planner.py compare flights_global.txt ICN SFO 08:00
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

