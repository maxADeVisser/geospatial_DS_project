from enum import Enum


class TimeFrequency(Enum):
    """A enum class to represent the different time frequencies for resampling the AIS dataset."""
    min_1 = "1min"
    min_10 = "10min"
    min_15 = "15min"
    min_30 = "30min"
    hour_1 = "1H"

class ShipType(Enum):
    """A enum class to represent the different ship types in the AIS dataset"""
    sailboat = 'Sailing'
    # 'Undefined'
    # 'Cargo'
    # 'Pleasure'
    # 'Passenger'
    # 'Fishing'
    # 'Other',
    # 'Pilot'
    # 'Law enforcement'
    # 'SAR'
    # 'Military'
    # 'Tug'
    # 'Dredging'
    # 'Towing'
    # 'HSC'
    # 'Tanker'
    # 'Not party to conflict'
    # 'Port tender'
    # 'Reserved'
    # 'WIG'
    # 'Diving'
    # 'Spare 1'
    # 'Anti-pollution'
    # 'Towing long/wide'
    # 'Medical'
    # 'Spare 2'
