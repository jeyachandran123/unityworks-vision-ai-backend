"""Seams to systems this application does not own.

Everything here talks to something outside the perception stack — a point of
sale, an ERP. Deliberately **not** in `vision_os/`: the platform is a perception
system, a till is not perception, and putting a POS adapter behind a vision port
would be the first crack in a boundary the whole architecture rests on.
"""
