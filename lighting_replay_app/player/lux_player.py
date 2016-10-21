"""
Play the light patterns for a specific room, using
the a Hidden Markov Model based on the prior data.
"""
# standard modules
import datetime
import time
import os.path
import argparse
import sys
import logging

# pip-installed packages
from sklearn.externals import joblib
from phue import Bridge

logger = logging.getLogger()


# local modules
from lux_time_utils import get_sunrise_sunset, time_of_day_to_zone,\
    dt_to_minutes, NUM_ZONES

def predict_hmm_by_zone(hmm_by_zone, dt_series):
    predictions = []
    last_zone = None
    last_cnt = None
    for dt in dt_series:
        (sunrise, sunset) = get_sunrise_sunset(dt.month, dt.day)
        zone = time_of_day_to_zone(dt_to_minutes(dt), sunrise, sunset)
        if zone != last_zone:
            if last_cnt is not None:
                (samples, states) = hmm_by_zone[last_zone].sample(last_cnt)
                predictions.extend([x[0] for x in samples])
            last_cnt = 1
        else:
            last_cnt += 1
        last_zone = zone
    if last_zone is not None:
        (samples, states) = hmm_by_zone[last_zone].sample(last_cnt)
        predictions.extend([x[0] for x in samples])
    assert len(predictions)==len(dt_series)
    return predictions

def dt_series_for_rest_of_day():
    """Return an array of datetimes starting one minute from now through
    11:59 today, at one minute intervals.
    """
    now = datetime.datetime.now()
    start_dt = now + datetime.timedelta(minutes=1)
    if start_dt.day!=now.day:
        # edge case: we are at the end of a day: return an empty list
        return []
    return \
        [datetime.datetime(year=start_dt.year, month=start_dt.month,
                           day=start_dt.day, hour=start_dt.hour,
                           minute=m)
         for m in range(start_dt.minute, 60)] + \
        [datetime.datetime(year=start_dt.year, month=start_dt.month,
                           day=start_dt.day, hour=h, minute=m)
         for (h, m) in [(h, m) for h in range(start_dt.hour+1, 24)
                        for m in range(0, 60)]]

def sleep_until(dt):
    """Sleep until the specified time
    """
    now = datetime.datetime.now()
    if now < dt:
        diff = dt - now
        secs = diff.days*24 + diff.seconds
        logger.info("Sleeping for %s seconds" % secs)
        time.sleep(secs)
    else:
        diff = now - dt
        if diff.days==0 and diff.seconds<=1:
            logger.info("No need to sleep until %s, already at %s" %
                        (dt, now))
            return
        else:
            raise Exception("%s is before the current time of %s!" % (dt, now))
    

def load_hmms(room):
    hmm_by_zone = []
    for zone in range(NUM_ZONES):
        fname = 'saved_hmms/hmm-%s-zone-%d.pkl' % (room, zone)
        if not os.path.exists(fname):
            raise Exception("Could not find file %s" % fname)
        hmm_by_zone.append(joblib.load(fname))
    return hmm_by_zone

def print_events_for_series(light_states, datetimes):
    current_state = None
    for (state, dt) in zip(light_states, datetimes):
        if state!=current_state:
            print("  Light set to %s at %s" %
                  ('On' if state else 'Off', dt))
        current_state = state

def build_predictions(rooms):
    """Build predictions for the specified rooms and return
    a sequence of event tuples of the form (datetime, room, light_state).
    """
    datetimes = dt_series_for_rest_of_day()
    events = []
    for room in rooms:
        hmm_by_zone = load_hmms(room)
        predictions = predict_hmm_by_zone(hmm_by_zone, datetimes)
        current_state = None
        for (state, dt) in zip(predictions, datetimes):
            if state!=current_state:
                events.append((dt, room, state),)
            current_state = state
    return sorted(events)

def control_light(bridge_ip, key, room_name, light_id, new_state,
                  dry_run=False):
    assert new_state in (True, False)
    logger.info("Setting light %s(%s) to %s" %
                (room_name, light_id, "ON" if new_state else "OFF"))
    if dry_run:
        return
    b = Bridge(bridge_ip, username=key)
    b.get_api()
    b.set_light(light_id, {'on':new_state})
    result = b.get_light(light_id)
    if result['state']['on']==new_state:
        logger.info("  Set light %s to %s" % (light_id, "ON" if new_state else "OFF"))
    else:
        logger.info("  Tried to set light %s to %s, but new state is %s" %
                    (light_id, new_state, result['state']['on']))

def test_lights(cycle_time, num_cycles, bridge_ip, key, rooms_to_light_ids, dry_run=False):
    rooms = sorted(rooms_to_light_ids.keys())
    for room in rooms:
        control_light(bridge_ip, key, room, rooms_to_light_ids[room],
                      False, dry_run=dry_run)
    logger.info("initially turned lights off")
    for room in rooms:
        logger.info("Testing light in room %s" % room)
        for i in range(num_cycles):
            logger.info("Starting test cycle %d for room %s" % (i, room))
            control_light(bridge_ip, key, room, rooms_to_light_ids[room],
                          True, dry_run=dry_run)
            time.sleep(cycle_time/2)
            control_light(bridge_ip, key, room, rooms_to_light_ids[room],
                          False, dry_run=dry_run)
            time.sleep(cycle_time/2)
    logger.info("Finished light testing")

    
def run(bridge_ip, key, rooms_to_light_ids, dry_run=False):
    rooms = sorted(rooms_to_light_ids.keys())
    test_lights(10, 6, bridge_ip, key, rooms_to_light_ids, dry_run)
    while True:
        events = build_predictions(rooms)
        if len(events)>0:
            today = events[0][0]
        else:
            logger.info("No events found for today, sleeping one minute")
            time.sleep(60)
            continue
        logger.info("%d events defined for %s" % (len(events), today))
        logger.debug("Events planned for %s" % today)
        for (dt, room, state) in events:
            logger.debug("  %s %s %s" % (dt, room, 'On' if state else 'Off'))

        for (dt, room, state) in events:
            sleep_until(dt)
            control_light(bridge_ip, key, room, rooms_to_light_ids[room],
                          True if state==1 else False, dry_run=dry_run)
        tomorrow = today + datetime.timedelta(days=1)
        tomorrow_dt = datetime.datetime(year=tomorrow.year,
                                        month=tomorrow.month,
                                        day=tomorrow.day,
                                        hour=0,
                                        minute=0)
        if tomorrow_dt > datetime.datetime.now():
            sleep_until(tomorrow_dt)

            

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Control a hue light')
    parser.add_argument('-d', '--dry-run', dest='dry_run',
                        action='store_true', default=False,
                        help="If specified, just print what would be done for today")
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help="If specified, print DEBUG level information")
    parser.add_argument('ip_address', metavar='IP_ADDRESS',
                        help="IP address for the bridge")
    parser.add_argument('key', metavar='KEY',
                        help="API key/username for the bridge")
    parser.add_argument('rooms', metavar='ROOM', type=str, nargs='+',
                        help="Rooms, in the form ROOM_NAME:LIGHT_ID")
    parsed_args = parser.parse_args(args)

    # set up room to light_id mapping
    light_ids = {}
    for room in parsed_args.rooms:
        room_info = room.split(':')
        if len(room_info)!=2:
            sys.stderr.write("Invalid format for a room: '%s', should be ROOM_NAME:LIGHT_ID\n" %
                             room)
            return 1
        light_ids[room_info[0]] = int(room_info[1])

    # set up logging
    if parsed_args.dry_run:
        formatter = logging.Formatter("%(asctime)s [DRY_RUN] %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(message)s")
    if parsed_args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    for ch in [logging.StreamHandler(),
               logging.FileHandler('lux_player.log', 'w')]:
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(level)
    
    try:
        run(parsed_args.ip_address, parsed_args.key, light_ids,
            dry_run=parsed_args.dry_run)
    except KeyboardInterrupt:
        logger.info("Got a keyboard interrupt, turning off lights and exiting")
        for (room_name, light_id) in light_ids.items():
            control_light(parsed_args.ip_address, parsed_args.key,
                          room_name, light_id, False,
                          dry_run=parsed_args.dry_run)        
    except Exception as e:
        logger.exception("Run aborted due to an exception: %s" % e)
    finally:
        logger.info("Exiting lux player")
    return 0
        
    

if __name__ == '__main__':
    sys.exit(main())

