import enum


class DTYPE(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1
    FLOAT8 = 2
    BFLOAT16 = 3


# FIXME(woosuk): This is hacky.
DTYPE.FLOAT32.size = 4
DTYPE.FLOAT16.size = 2
DTYPE.FLOAT8.size = 1
DTYPE.BFLOAT16.size = 2

_DTYPE_REGISTRY = {
    # Float32
    "float32": DTYPE.FLOAT32,
    "float": DTYPE.FLOAT32,
    # Float16
    "float16": DTYPE.FLOAT16,
    "half": DTYPE.FLOAT16,
    # Float8
    "float8": DTYPE.FLOAT8,
    # BFloat16
    "bfloat16": DTYPE.BFLOAT16,
}


def get_dtype(dtype: str) -> DTYPE:
    if dtype.lower() not in _DTYPE_REGISTRY:
        raise ValueError(f"Unknown dtype {dtype}")
    return _DTYPE_REGISTRY[dtype.lower()]


def dtype_to_str(dtype: DTYPE) -> str:
    if dtype == DTYPE.FLOAT32:
        return "float"
    elif dtype == DTYPE.FLOAT16:
        return "half"
    elif dtype == DTYPE.FLOAT8:
        return "float8"
    elif dtype == DTYPE.BFLOAT16:
        return "bfloat16"
    else:
        raise ValueError(f"Unknown dtype {dtype}")


import enum


class CommType(enum.Enum):
    AllReduce = enum.auto()
    AllGather = enum.auto()
    AllToAll = enum.auto()
    ReduceScatter = enum.auto()


_STR_TO_COMM_TYPE = {
    "AllReduce": CommType.AllReduce,
    "AllGather": CommType.AllGather,
    "AllToAll": CommType.AllToAll,
    "ReduceScatter": CommType.ReduceScatter,
}


def _str_to_comm_type(comm_type: str) -> CommType:
    if comm_type not in _STR_TO_COMM_TYPE:
        raise ValueError(f"Invalid comm_type: {comm_type}")
    return _STR_TO_COMM_TYPE[comm_type]


class CollectiveComm:

    def __init__(
        self,
        comm_type: str,
        num_devices: int,
        size_factor: float = 1.0,
    ) -> None:
        self.comm_type = _str_to_comm_type(comm_type)
        self.num_devices = num_devices
        self.size_factor = size_factor

    def __repr__(self) -> str:
        if self.size_factor == 1:
            return (
                f"CollectiveComm(comm_type={self.comm_type}, "
                f"num_devices={self.num_devices})"
            )
        else:
            return (
                f"CollectiveComm(comm_type={self.comm_type}, "
                f"num_devices={self.num_devices}, "
                f"size_factor={self.size_factor})"
            )
