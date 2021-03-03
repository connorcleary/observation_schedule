import numpy as np
from skyfield.api import Topos, load
from skyfield.toposlib import wgs84
import datetime as dt
from pytz import timezone
from skyfield import almanac
import os


def main():
    """
    Writes an observation schedule to a text file.

        Parameters:
        Returns:
            none
    """
    config_fn = input("please enter configuration filename: ").strip('.txt')
    t0, targets, longitude, latitude, time_zone = get_config_data(config_fn+'.txt')

    window = [[t0[0], t0[1], t0[2] + 0.5], [t0[0], t0[1], t0[2] + 1.5]]
    location = [latitude, longitude]

    target_satellites = get_target_satellites(targets)
    t, events = find_events(target_satellites, location, window, time_zone)
    tt = convert_times(t)
    tt_series = time_series(tt, events)
    aos, los = find_sunlit_events(target_satellites, tt_series)
    dark_aos, dark_los = find_night_events(aos, los, location,
                                           time_zone, window)
    events_ordered = ordered_events(dark_aos, dark_los,
                                    target_satellites).tolist()
    events_positions = ccd_schedule(events_ordered, location)

    write_to_file(events_positions, time_zone)
    print("finished")


def get_config_data(filename):
    with open("%s" % filename, "r") as f:
        t0 = [int(x) for x in f.readline().split(',')]
        targets = f.readline().split(';')
        longitude = f.readline()
        latitude = f.readline()
        time_zone = f.readline().strip('\n')

    return t0, targets, longitude, latitude, time_zone


def get_data():
    """
    Downloads satellite data and creates a list of satellite objects

        Parameters:
        Returns:
            satellites: array of skyfield satellite objects
    """
    # download satellite data
    with open("urls.txt", "r") as f:
        satellites = list()
        for line in f.readlines():
            # you could switch this appendage around if you wanted
            # to have your sources ranked in a reverse priority
            satellites = satellites + load.tle_file(line.replace('\n', ''))
            os.remove(line.replace('\n', '').split('/')[-1])
    return satellites


def get_target_satellites(targets):
    """
    Create array of satellites in the config file

        Parameters:
        Returns:
            target_satellites: array of target satellite objects
    """

    satellites = get_data()
    target_satellites = list()
    # search for each satellite by satcat
    for target in targets:
        for satellite in satellites:
            if "#" + target + ' ' in satellite.target_name:
                target_satellites = target_satellites + [satellite]
                break

    return target_satellites


def convert_times(t):
    """
    Converts time object to terrestrial time floating point

        Parameters:
            t: array of time objects
        Returns:
            tt: array of floating point terrestrial times
    """
    tt = []
    for ti in t:
        tt.append(ti.tt)

    return tt


def time_series(t, events):
    """
    Creates an array of linspaces for the period of each satellite pass

        Parameters:
            t:  array of terrestrial times for each event
            events: array of pass events
        Return:
            t_series: array of time linspaces for each event
    """
    t_series = []
    for idx, (ti, event) in enumerate(zip(t, events)):
        t_series.append([])
        counter = 0
        while counter < len(ti) - 2:
            t_series[idx].append(np.linspace(ti[counter], ti[counter + 2
                                             - event[counter]], 1000))
            counter = counter + 3 - event[counter]
    return t_series


def find_events(target_satellites, location, window, time_zone):
    """
    Finds all events for each target satellite

        Parameters:
            target_satellites: list of target satellite objects
            location: site coordinates
            window: time window to look for events in
            time_zone: time zone name
        Returns:
            events: array of events, consisting of a list over events for each
                    satellite. see skyfield for event notation
            t: array of time objects corresponding to each event
    """
    # set timezone and window in correct time format
    midday = dt.datetime(window[0][0], window[0][1], int(np.floor(window[0][2])),
                         12, tzinfo = timezone(time_zone))
    next_midday = midday + dt.timedelta(days=1)
    ts = load.timescale()
    t0 = ts.from_datetime(midday)
    t1 = ts.from_datetime(next_midday)

    # set location and initialize arrays
    sight = Topos(location[0].strip('-') + 'S', location[1] + 'E')
    events = [None] * len(target_satellites)
    t = [None] * len(target_satellites)

    # find events for each satellite
    for idx, satellite in enumerate(target_satellites):
        t[idx], events[idx] = satellite.find_events(sight, t0, t1,
                                                    altitude_degrees=20.0)

    return t, events


def find_sunlit_events(target_satellites, t):
    """
    Reduce events to the time for which they are sunlit,

        Parameters:
            target_satellites: list of target satellite objects
            t: list of event times for each satellite
        Returns:
            aos: array of acquisition of signal times,
                 list of aos for each satellite
            los: array of loss of signal times,
                 list of los for each satellite
    """
    eph = load('de421.bsp')  # load ephemeris - may need to add a reload here
    aos = []
    los = []
    ts = load.timescale()

    # for each satellite
    for idx, (satellite_t, satellite) in enumerate(zip(t, target_satellites)):
        satellite_aos = []
        satellite_los = []

        for pass_dx, pass_times in enumerate(satellite_t):
            start_id = 1001
            end_id = -1

            # find period of sunlight on each pass, this could be optimized
            for tdx, time in enumerate(pass_times):
                if satellite.at(ts.tt_jd(time)).is_sunlit(eph):
                    start_id = tdx
                    break

            pass_times = np.flipud(pass_times)
            for tdx, time in enumerate(pass_times):
                if satellite.at(ts.tt_jd(time)).is_sunlit(eph):
                    end_id = 999 - tdx
                    break
            pass_times = np.flipud(pass_times)
            # if event has any sunlit: here the length of sunlight pass
            # required could be implemeneted
            if start_id < 1001 and end_id > -1:
                satellite_aos.append(pass_times[start_id])
                satellite_los.append(pass_times[end_id])

        aos.append(satellite_aos)
        los.append(satellite_los)

    return aos, los


def find_night_events(aos, los, location, time_zone, window):
    """
    Reduce events in the form of aos and los to events which occur at night

        Parameters:
            aos: array of aos for each satellite
            los: arrau of los for each satellite
            location: coordinates for observing
            time_zone: time_zone name
            window: observation window
        Returns:
            dark_aos: array of aos for events which occur at night
            dark_los: array of los for events which occur at night
    """

    # this is all required to find astronomical events
    midday = dt.datetime(window[0][0], window[0][1], int(np.floor(window[0][2])),
                         12, tzinfo = timezone(time_zone))
    next_midday = midday + dt.timedelta(days=1)

    ts = load.timescale()
    t0 = ts.from_datetime(midday)
    t1 = ts.from_datetime(next_midday)
    eph = load('de421.bsp')
    sight = Topos(location[0].strip('-') + 'S', location[1] + 'E')

    # load night conditions
    f = almanac.dark_twilight_day(eph, sight)
    # find dawn/dusk events
    astro_t, astro_events = almanac.find_discrete(t0, t1, f)

    astro_dusk = astro_t[2]
    astro_dawn = astro_t[4]

    dark_aos = []
    dark_los = []

    # find events which fall within dawn/dusk
    for satellite_aos, satellite_los in zip(aos, los):
        dark_start = []
        dark_end = []
        for start, end in zip(satellite_aos, satellite_los):
            if (astro_dusk.tt <= start <= astro_dawn.tt) \
            and (astro_dusk.tt <= end <= astro_dawn.tt):
                dark_start.append(start)
                dark_end.append(end)
        dark_aos.append(dark_start)
        dark_los.append(dark_end)

    return dark_aos, dark_los


def ordered_events(aos_list, los_list, target_satellites):
    """
    Creates a list of ordered events, which may be overlapping

        Parameters:
            aos_list: array of aos
            los_list: array of los
            target_satellites:
        Returns:
            events: array of events in the format [satellite, aos, los]
    """
    events = []

    for satellite, aos, los in zip(target_satellites, aos_list, los_list):
        for start, end in zip(aos, los): events.append([satellite, start, end])

    events = np.array(events)
    events = events[events[:, 1].argsort()]

    return events


def calendar_date(schedule):
    """
    Convert schedule times to calender date times as required in mount control
    software

        Parameters:
            schedule: schedule array with terrestrial times
        Returns:
            schedule: schedule array with calender times
    """
    ts = load.timescale()

    # add the calender date to the observation schedule
    for i in range(len(schedule)):
        schedule[i][1] = ts.tt_jd(schedule[i][1]).tt_calendar()
        schedule[i][2] = ts.tt_jd(schedule[i][2]).tt_calendar()

    return schedule


def ccd_schedule(schedule, location):
    """
    Creates a schedule for use with and external
    Parameters:
        schedule:
        location:
    Returns:

    """
    # user inputs
    pass_T = 120.  # length of minimum pass in seconds
    exposure_T = 20.  # length of exposure
    buffer_T = 40. + exposure_T  # time between passes

    # predefinition
    ts = load.timescale()
    site = wgs84.latlon(float(location[0]), float(location[1]))
    ccd_schedule = []
    jSec = 1.0 / 24 / 60 / 60  # second in julian time

    # add first event, assuming first event is long enough
    event = schedule[0]
    difference = event[0] - site
    ccd_schedule.append([event[0], event[1], event[1] + buffer_T * jSec])
    ccd_schedule[0].extend([difference.at(ts.tt_jd(ccd_schedule[0][1]
                                          + exposure_T / 2. * jSec)),
                            difference.at(ts.tt_jd(ccd_schedule[0][2]
                                          + exposure_T / 2. * jSec))])

    for event in schedule[1:]:
        if event[1] - buffer_T * jSec > ccd_schedule[-1][2] \
        and event[2] > ccd_schedule[-1][2] + (buffer_T + pass_T) * jSec:
            difference = event[0] - site
            ccd_schedule.append([event[0], max(ccd_schedule[-1][2]
                                               + buffer_T*jSec, event[1])])
            ccd_schedule[-1].append(ccd_schedule[-1][1] + buffer_T * jSec)
            ccd_schedule[-1].extend([difference.at(ts.tt_jd(ccd_schedule[-1][1]
                                                            + exposure_T / 2.*jSec)),
                                     difference.at(ts.tt_jd(ccd_schedule[-1][2]
                                                            + exposure_T / 2.*jSec))])

    return ccd_schedule


def greatest_elevation_schedule(ordered_events, location):

    pass_T = 120.  # length of minimum pass in seconds
    exposure_T = 20.  # length of exposure
    buffer_T = 40. + exposure_T  # time between exposures

    ts = load.timescale()
    site = wgs84.latlon(location[0], location[1], elevation_m=0)
    ccd_schedule = []
    jSec = 1.0 / 24 / 60 / 60  # second in julian time

    for event in ordered_events:
        elevations = list()
        for tdx, t in enumerate(np.linspace(event[1], event[2], 100)):
            difference = event[0] - site
            elevations.append(difference.at(ts.tt_jd(tdx)).elevation)

        event.append(event[1]
                     + (event[2]-event[1])/100*elevations.index(max(elevations)))


def write_to_file(schedule, zone):
    """
    Write schedule to a text file

        Parameters:
            schedule: array of observable events
            zone: time zone name
        Returns:
            none
    """
    from pytz import timezone
    ts = load.timescale()
    date = ts.tt_jd(schedule[0][1]).astimezone(timezone(zone))
    if date.hour < 12:
        f = open(str(date.year) + '-' + str(date.month) + '-'
                 + str(date.day - 1) + '.txt', "w")
        f.write(str(date.year) + '-' + str(date.month) + '-'
                + str(date.day - 1) + ',' + str(len(schedule)) + '\n')
    else:
        f = open(str(date.year) + '-' + str(date.month) + '-'
                 + str(date.day) + '.txt', "w")
        f.write(str(date.year) + '-' + str(date.month) + '-'
                + str(date.day) + ',' + str(len(schedule)) + '\n')
    for event in schedule:
        aos = ts.tt_jd(event[1]).astimezone(timezone(zone))
        los = ts.tt_jd(event[2]).astimezone(timezone(zone))
        f.write(
            event[0].name + ',' + str(aos.hour) + ':' + str(aos.minute)
            + ':' + str(aos.second) + ',' + str(los.hour) + ':'
            + str(los.minute) + ':' + str(los.second) + ','
            + str(event[3].radec()[0]) + ',' + str(event[3].radec()[1]) + ','
            + str(event[4].radec()[0]) + ',' + str(event[4].radec()[1]) + '\n'
            )
    f.close()


if __name__ == "__main__":
    main()
