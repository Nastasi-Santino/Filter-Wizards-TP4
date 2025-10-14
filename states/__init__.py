from . import state1, state2, state3, state4

HANDLERS = {
    1: state1.run,
    2: state2.run,
    3: state3.run,
    4: state4.run,
}

__all__ = ["HANDLERS"]