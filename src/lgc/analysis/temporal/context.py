"""Construction of month-level reception contexts and training windows."""

from collections.abc import Iterable

import numpy as np

from ...model import LinkGraph
from .models import CombinedWindow, TemporalContext


def build_temporal_context(graph: LinkGraph) -> TemporalContext:
    months = sorted(graph.all_packets["month"].dropna().unique().tolist())
    month_values = graph.all_packets["month"].to_numpy()
    packet_rows_by_month = {
        month: np.flatnonzero(month_values == month) for month in months
    }

    sensor_total_by_month: dict[str, dict[str, int]] = {}
    total_packets_by_month: dict[str, int] = {}
    for month in months:
        packets = graph.all_packets[graph.all_packets["month"] == month]
        counts = packets.groupby("sensor").size()
        sensor_total_by_month[month] = {
            str(sensor): int(count) for sensor, count in counts.items()
        }
        total_packets_by_month[month] = int(len(packets))

    packet_gateway_sets = (
        graph.df.groupby(["sensor", "counter"])["gateway"]
        .agg(lambda values: frozenset(values))
        .reset_index(name="gw_set")
        .merge(
            graph.all_packets[["sensor", "counter", "month"]],
            on=["sensor", "counter"],
            how="left",
        )
    )
    pattern_counts = (
        packet_gateway_sets.groupby(["month", "sensor", "gw_set"])
        .size()
        .reset_index(name="n_packets")
    )

    pattern_lookup_by_month = {
        month: {
            sensor: [
                (row.gw_set, int(row.n_packets))
                for row in pattern_counts[
                    (pattern_counts["month"] == month)
                    & (pattern_counts["sensor"] == sensor)
                ].itertuples(index=False)
            ]
            for sensor in graph.all_sensors
        }
        for month in months
    }

    return TemporalContext(
        months=months,
        packet_rows_by_month=packet_rows_by_month,
        sensor_total_by_month=sensor_total_by_month,
        total_packets_by_month=total_packets_by_month,
        pattern_lookup_by_month=pattern_lookup_by_month,
    )


def combine_window(
    graph: LinkGraph,
    context: TemporalContext,
    months: Iterable[str],
) -> CombinedWindow:
    month_list = list(months)
    if not month_list:
        raise ValueError("Training window must contain at least one month.")

    unknown = [month for month in month_list if month not in context.packet_rows_by_month]
    if unknown:
        raise KeyError(f"Unknown month(s): {unknown}")

    rows = np.concatenate([context.packet_rows_by_month[month] for month in month_list])
    rows.sort()

    sensor_total = {
        sensor: sum(
            context.sensor_total_by_month[month].get(sensor, 0)
            for month in month_list
        )
        for sensor in graph.all_sensors
    }
    sensors_active = [
        sensor for sensor in graph.all_sensors if sensor_total[sensor] > 0
    ]

    combined_counts: dict[str, dict[frozenset[str], int]] = {
        sensor: {} for sensor in graph.all_sensors
    }
    for month in month_list:
        for sensor, patterns in context.pattern_lookup_by_month[month].items():
            for gateway_set, count in patterns:
                counts = combined_counts[sensor]
                counts[gateway_set] = counts.get(gateway_set, 0) + int(count)

    pattern_lookup = {
        sensor: list(combined_counts[sensor].items()) for sensor in graph.all_sensors
    }

    return CombinedWindow(
        months=month_list,
        rows=rows,
        packet_matrix=graph.incidence[rows],
        packet_sensor=graph.packet_sensor[rows],
        sensor_total=sensor_total,
        sensors_active=sensors_active,
        pattern_lookup=pattern_lookup,
        total_packets=int(len(rows)),
    )
