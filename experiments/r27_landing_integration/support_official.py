"""Authentic official comparison adapter for the frozen support-aware source."""
from experiments.r27_landing_integration import official


official.FROZEN = {
    "support_plan.json": "5e88e5fd8f050e76eac2a75207452184f0705a4ad7833073444c9a877542c3b5",
    "model.json": "447660558c5ba67ea65e53be568e493bdd5f42532b558f440884b3fb610b5939",
    "support_aware.py": "e40865f5c78851fb02f361d0cac9709f338e3a2ae016e71575e57bb111ee2bc3",
}
_score = official.score


def score(item):
    name, detector, folder, env = item
    if name == "r27":
        detector = "experiments.r27_landing_integration.support_aware"
    return _score((name, detector, folder, env))


official.score = score


if __name__ == "__main__":
    official.main()
