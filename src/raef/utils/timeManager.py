from datetime import datetime, timedelta

class TimeManager(object):
    """
    Class to handle time
    """
    def getPeriodSeconds(periodicity : dict):
        """
        Get periodicity in seconds
        """
        periodicitySeconds : int = 0
        for key in periodicity:
            if key == "sec":
                periodicitySeconds += periodicity[key]
            elif key == "min":
                periodicitySeconds += periodicity[key] * 60
            elif key == "hour":
                periodicitySeconds += periodicity[key] * 60 * 60
            elif key == "day":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24
            elif key == "week":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * 7
            elif key == "month":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * (365 / 12)
            elif key == "year":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * 365

        return periodicitySeconds
    
    def timeDiffSeconds(startStr : str, endStr : str, timestampFormart : str) -> int:
        """
        Method to get timediff in seconds
        """
        try:
            return abs(int((
                datetime.strptime(startStr, timestampFormart) - datetime.strptime(endStr, timestampFormart)
            ).total_seconds()))
        except TypeError as e:
            return 60

    def nextTimeStamp(timestamp : str, formatTime : str, period : int) -> str:
        """
        Method to add period to timestamp to generate next timestamp
        """
        return (datetime.strptime(timestamp, formatTime) + timedelta(seconds=period)).strftime(formatTime)
    
    def convertTimeFormat(timestamp : str, originalFormat : str, targetFormat : str) -> str:
        """
        Method to conver time stamp
        """
        try:
            dt : datetime = datetime.strptime(timestamp, originalFormat)
            return dt.strftime(targetFormat)
        except TypeError as e:
            return datetime.now().strftime(targetFormat)
        except ValueError as e:
            raise Exception(f"Invalid date format: {timestamp} - {e}")

