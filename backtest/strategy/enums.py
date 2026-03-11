from enum import Enum, auto


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class SizeType(Enum):
    CONTRACTS = auto()
    DOLLARS = auto()
    PCT_RISK = auto()


class ExitReason(Enum):
    SL = auto()
    TP = auto()
    EOD = auto()
    SIGNAL = auto()
    SAME_BAR_SL = auto()
    SAME_BAR_TP = auto()
    FORCED_EXIT = auto()