# Project 3 — Flight Route & Fare Comparator (FlyWise)

This program loads flight schedules, builds a directed graph of routes, and compares the earliest-arrival and cheapest itineraries between two airports. It supports economy, business, and first-class fares and enforces a minimum layover time between connecting flights.

All logic is implemented using adjacency lists, Dijkstra-based search, and structured itinerary objects.

---

# Files in This Project

- flight.py — Flight dataclass (origin, destination, code, times, fares)  
- itinerary.py — Itinerary class (stores flights and totals)  
- graph.py — build_graph and the search algorithms  
- flight_planner.py — CLI entry point  
- tests/ — 26 auto-grading tests  

---

# How to Run the Program

## Run the CLI

From the project root:

```bash
python flight_planner.py compare <file> <ORIGIN> <DEST> <HH:MM>
```

# Example
```bash
python flight_planner.py compare data/sample1.txt ICN LAX 08:00
```

# Output Format

The CLI prints a table such as:

==============================================
Earliest-arrival Itinerary
----------------------------------------------
Depart: ICN 08:00
Arrive: LAX 19:43
Duration: 11h 43m
Stops: 1
Price: 850 (economy)

Cheapest Itinerary
----------------------------------------------
Depart: ICN 09:00
Arrive: LAX 21:30
Duration: 12h 30m
Stops: 2
Price: 620 (economy)
==============================================

# Input File Format
Each line in the input schedule file:
```bash
ORIGIN DEST CODE DEPART ARRIVE ECO BUS FIRST
Example: ICN LAX OZ201 08:00 18:30 850 1400 2400

Malformed lines are ignored.
```

# Graph Design
The system uses a directed adjacency-list graph:
```bash
flights_from: dict[str, list[Flight]]
```
Key = origin airport
Value = list of Flight objects departing from that airport
Only airports with outgoing flights appear as keys (required by tests)

# Search Algorithms

```bash
# Earliest-Arrival (Time-Based)

Modified Dijkstra where:

Distance = earliest possible arrival time

Connection valid only if layover ≥ MIN_LAYOVER_MINUTES

# Cheapest Route (Fare-Based)

Also Dijkstra:

Distance = total fare (economy/business/first)

Still must obey departure time + layover rules

Both algorithms track:

best_time or best_cost

previous for path reconstruction
```

# Complexity Analysis
```bash
Let:

V = number of airports

E = number of flights

# Loading Flights
Time:  O(n)

# build_graph(flights)
Time:  O(n)
Space: O(n)

# Earliest-arrival search (Dijkstra)
Time:  O(E log V)
Space: O(E + V)

# Cheapest-fare search (Dijkstra)

# Same complexity:

Time:  O(E log V)
Space: O(E + V)
```

# Edge Cases

Malformed input lines → ignored

Unknown airport codes → printed as errors

No possible itinerary → user sees a clear message

Layovers < MIN_LAYOVER_MINUTES → connection rejected

Airports with no outgoing flights are excluded from graph keys

# Itinerary Object

The Itinerary class:

Stores list of flights.
Computes total duration.
Computes total cost based on cabin.
Formats output for CLI.

# Running Tests
```bash
# Run the full test suite:
python -m pytest -q

# Expected result:
26 passed
```

# Assumptions

All times use HH:MM 24-hour format

No timezone conversions needed

Test data contains valid arrival > departure

Layover rules apply only between connecting flights

Airport codes are uppercase

# Summary
This project implements:

A clean adjacency-list flight graph.

Two Dijkstra-based search algorithms.

A structured Itinerary class.

A full command-line comparison tool.

Complete test coverage (26/26 passed).