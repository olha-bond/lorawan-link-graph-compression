import pandas as pd

from lgc.model import LinkGraph


def test_packet_link_words_support_more_than_32_links():
    rows = []
    for sensor in ["s1", "s2"]:
        for gateway_index in range(20):
            gateway = f"g{gateway_index:02d}"
            rows.append(
                {
                    "sensor": sensor,
                    "gateway": gateway,
                    "counter": gateway_index,
                    "month": "2024-01",
                    "link_id": f"{sensor}→{gateway}",
                    "rssi": -90.0,
                    "snr": 5.0,
                }
            )

    graph = LinkGraph(pd.DataFrame(rows))
    assert graph.n_links == 40
    assert graph.packet_link_words.shape[1] == 2
