"""In-memory representation of packet receptions and sensor--gateway links."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LinkGraph:
    df: pd.DataFrame

    all_packets: pd.DataFrame = field(init=False)
    total_unique_packets: int = field(init=False)
    packet_sensor: np.ndarray = field(init=False)

    all_link_ids: list[str] = field(init=False)
    all_sensors: list[str] = field(init=False)
    all_gateways: list[str] = field(init=False)
    link_to_col: dict[str, int] = field(init=False)
    link_sensor: dict[str, str] = field(init=False)

    sensor_total: dict[str, int] = field(init=False)
    sensor_packet_indices: dict[str, np.ndarray] = field(init=False)
    sensor_packet_count: dict[str, int] = field(init=False)
    sensor_gateway_lists: dict[str, list[str]] = field(init=False)

    incidence: np.ndarray = field(init=False)
    link_local_mask: dict[str, np.ndarray] = field(init=False)
    packet_link_words: np.ndarray = field(init=False)
    pattern_lookup: dict[str, list[tuple[frozenset[str], int]]] = field(init=False)

    def __post_init__(self) -> None:
        self._build_packet_index()
        self._build_link_index()
        self._build_incidence()
        self._build_per_sensor_masks()
        self._build_packet_link_words()
        self._build_pattern_lookup()

    def _build_packet_index(self) -> None:
        packets = (
            self.df.groupby(["sensor", "counter"])
            .size()
            .rename("n_rows")
            .reset_index()[["sensor", "counter"]]
        )
        packet_month = (
            self.df.groupby(["sensor", "counter"])["month"].min().reset_index()
        )
        self.all_packets = packets.merge(
            packet_month,
            on=["sensor", "counter"],
            how="left",
        )
        self.total_unique_packets = len(self.all_packets)
        self.packet_sensor = self.all_packets["sensor"].to_numpy()

    def _build_link_index(self) -> None:
        self.all_link_ids = sorted(self.df["link_id"].unique())
        self.all_sensors = sorted(self.df["sensor"].unique())
        self.all_gateways = sorted(self.df["gateway"].unique())
        self.link_to_col = {
            link: column for column, link in enumerate(self.all_link_ids)
        }
        self.link_sensor = {
            link: link.split("→", maxsplit=1)[0] for link in self.all_link_ids
        }

        sensor_totals = self.all_packets.groupby("sensor").size()
        self.sensor_total = {
            str(sensor): int(total) for sensor, total in sensor_totals.items()
        }
        self.sensor_gateway_lists = {
            str(sensor): sorted(gateways)
            for sensor, gateways in self.df.groupby("sensor")["gateway"].unique().items()
        }

    def _build_incidence(self) -> None:
        packet_keys = list(zip(self.all_packets["sensor"], self.all_packets["counter"]))
        packet_index = {key: index for index, key in enumerate(packet_keys)}

        incidence = np.zeros(
            (len(packet_index), len(self.all_link_ids)),
            dtype=bool,
        )
        row_index = np.fromiter(
            (
                packet_index.get(key, -1)
                for key in zip(self.df["sensor"], self.df["counter"])
            ),
            dtype=np.int64,
            count=len(self.df),
        )
        column_index = (
            self.df["link_id"]
            .map(self.link_to_col)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        valid = (row_index >= 0) & (column_index >= 0)
        incidence[row_index[valid], column_index[valid]] = True
        self.incidence = incidence

    def _build_per_sensor_masks(self) -> None:
        self.sensor_packet_indices = {
            sensor: np.flatnonzero(self.packet_sensor == sensor)
            for sensor in self.all_sensors
        }
        self.sensor_packet_count = {
            sensor: int(len(indices))
            for sensor, indices in self.sensor_packet_indices.items()
        }
        self.link_local_mask = {}
        for column, link in enumerate(self.all_link_ids):
            sensor = self.link_sensor[link]
            indices = self.sensor_packet_indices[sensor]
            self.link_local_mask[link] = self.incidence[indices, column].copy()

    def _build_packet_link_words(self) -> None:
        # multi-word: graphs wider than one uint32
        word_size = 32
        n_words = (self.n_links + word_size - 1) // word_size
        words = np.zeros(
            (self.total_unique_packets, n_words),
            dtype=np.uint32,
        )
        for column in range(self.n_links):
            word_index, bit_index = divmod(column, word_size)
            words[self.incidence[:, column], word_index] |= np.uint32(1 << bit_index)
        self.packet_link_words = words

    def _build_pattern_lookup(self) -> None:
        gateway_sets = (
            self.df.groupby(["sensor", "counter"])["gateway"]
            .agg(lambda values: frozenset(values))
            .reset_index(name="gw_set")
        )
        counts = (
            gateway_sets.groupby(["sensor", "gw_set"])
            .size()
            .reset_index(name="n_packets")
        )
        self.pattern_lookup = {
            sensor: [
                (gateway_set, int(n_packets))
                for gateway_set, n_packets in counts.loc[
                    counts["sensor"] == sensor,
                    ["gw_set", "n_packets"],
                ].itertuples(index=False, name=None)
            ]
            for sensor in self.all_sensors
        }

    @property
    def n_links(self) -> int:
        return len(self.all_link_ids)

    def __repr__(self) -> str:
        return (
            f"LinkGraph(sensors={len(self.all_sensors)}, "
            f"gateways={len(self.all_gateways)}, "
            f"links={self.n_links}, packets={self.total_unique_packets})"
        )
