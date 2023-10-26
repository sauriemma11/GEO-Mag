import numpy as np
import datetime as dt


def calculate_time_difference(longitude_degrees, hemisphere='W'):
    # Calculate the time difference for the input longitude
    # Input should be in degrees WEST. If east, (GK2A for example is at
    # 128.2 E) input degrees east + 360
    if hemisphere == 'E':
        longitude_degrees = 360 - longitude_degrees

    time_diff = (longitude_degrees / 360) * 24

    return time_diff


def calculate_total_magnetic_field(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def find_noon_and_midnight_time(time_diff, date_str, gk2a=False):
    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    date_obj_previous_day = date_obj - dt.timedelta(
        days=1)  # For plotting noon time GK2A

    midnight_time_UT = dt.datetime(date_obj.year, date_obj.month, date_obj.day,
                                   0, 0)
    noon_time_UT = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 12,
                               0)

    if gk2a is True:
        noon_time = dt.datetime(date_obj_previous_day.year,
                                date_obj_previous_day.month,
                                date_obj_previous_day.day, 12, 0)
        noon_time = noon_time + dt.timedelta(hours=time_diff)
        midnight_time = midnight_time_UT + dt.timedelta(hours=time_diff)
    else:
        midnight_time = midnight_time_UT + dt.timedelta(hours=time_diff)
        noon_time = noon_time_UT + dt.timedelta(hours=time_diff)

    return noon_time, midnight_time

# print(find_noon_and_midnight_time(calculate_time_difference(128.2),
# '2022-08-15', gk2a=True))
