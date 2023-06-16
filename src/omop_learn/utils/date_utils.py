import pandas as pd


REFTIME = pd.Timestamp("1900-01-01")
def to_unixtime(str_time_array):
    unix_times = (pd.to_datetime(str_time_array) - REFTIME) // pd.Timedelta("1d")
    return unix_times


def from_unixtime(unix_time_array):
    datetimes = [pd.to_datetime(t * pd.Timedelta("1d") + REFTIME) for t in unix_time_array]
    return datetimes
