from pathlib import Path


class Paths:
    def __init__(self, data_root, out_dir):
        self.data_root = data_root
        self.out_dir = out_dir

    @property
    def metadata(self):
        return (
            self.data_root
            / "dataset"
            / "lorawan_metadata"
            / "lorawan_combined_dataset.parquet"
        )

    @classmethod
    def resolve(cls, data_root, out_dir):
        data_root = Path(data_root).expanduser().resolve()
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(data_root=data_root, out_dir=out_dir)


EXPECTED_SENSORS = 10
EXPECTED_GATEWAYS = 3
EXPECTED_LINKS = EXPECTED_SENSORS * EXPECTED_GATEWAYS

REQUIREMENTS = [(90, 80), (95, 85), (98, 90)]

K_GRID = [3, 5, 7, 10, 12, 15, 20, 25, 30]

STABILITY_REQUIREMENT = (95, 85)

LAMBDA_MAIN = 1.0

COARSE_LAMBDAS = [0.0, 0.25, 0.5, 1.0]
FINE_LAMBDAS = [0.0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.25, 0.5, 1.0]

REF_KS = [10, 12, 16]

GRID_P_MIN = list(range(80, 100))
GRID_S_MIN = list(range(50, 100))

TEMPORAL_FROZEN_INITIAL_MONTHS = 3
