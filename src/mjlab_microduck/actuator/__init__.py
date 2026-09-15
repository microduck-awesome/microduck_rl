"""Actuator exports, loaded lazily so CPU model loading avoids mjlab plugins."""

__all__ = [
    "BacklashEncoderBamActuator",
    "BacklashEncoderBamActuatorCfg",
    "FrictionDRBamActuator",
    "FrictionDRBamActuatorCfg",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import friction_dr_bam
    value = getattr(friction_dr_bam, name)
    globals()[name] = value
    return value
