"""
Time utilities for lux analysis and replay
"""
import datetime

import astral # use pip install to get this

# Sunrise sunset data for Sunnyvale, CA
# TODO: change this to call some kind of service to find my current location

LOCATION = astral.Location(('Sunnyvale', 'USA', 37.3643427, -122.0080058, 'US/Pacific'),)
YEAR = 2017

def get_sunrise_sunset(month, day):
    s = LOCATION.sun(datetime.date(YEAR, month, day))
    def dt_to_mins(dt):
        return dt.hour*60+dt.minute
    return (dt_to_mins(s['sunrise']), dt_to_mins(s['sunset']))

        

# # We divide the day into "zones" based on a rough idea of the amount of sunlight.
def time_of_day_to_zone(minutes, sunrise, sunset):
    if minutes < sunrise:
        return 0 # early morning
    elif minutes <= (sunset-30):
        return 1 # daytime
    elif minutes <= max(sunset+60,21.5*60):
        return 2 # evening
    else:
        return 3 # later evening
    
NUM_ZONES=4

def dt_to_minutes(dt):
    return dt.time().hour*60 + dt.time().minute

def minutes_to_time(minutes):
    hrs = int(minutes/60)
    mins = minutes-(hrs*60)
    assert mins<60
    return (hrs, mins)

