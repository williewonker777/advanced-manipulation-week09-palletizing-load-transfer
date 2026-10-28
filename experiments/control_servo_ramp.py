from experiments.d2_servo import Controller as Base
class Controller(Base):
    USE_C3 = False
    LOAD_SOURCE = "ramp"
