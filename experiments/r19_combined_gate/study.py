"""Run the unchanged R18 integrated scorer with the immutable R19 V2 policy."""
from experiments.r18_combined_gate import study as frozen
from experiments.r19_combined_gate.policy import Controller, Detector


def main():
    frozen.Controller = Controller
    frozen.Detector = Detector
    frozen.main()


if __name__ == "__main__":
    main()
