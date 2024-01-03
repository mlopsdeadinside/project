import pandas as pd

def proc_data(info):
    info = info.drop_duplicates() # Drop duplicates
    info = info.dropna() # Drop null values
    info = pd.get_dummies(info, columns=['Machine_ID', 'Sensor_ID'])
    info['Timestamp'] = pd.to_datetime(info['Timestamp']) # Convert to datetime
    info['Hour'] = info['Timestamp'].dt.hour # Extract hour
    info['DayInWeek'] = info['Timestamp'].dt.dayofweek # Extract day in week
    info['Month'] = info['Timestamp'].dt.month # Extract month
    return info