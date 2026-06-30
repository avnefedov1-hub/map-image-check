"""Tests for AD remote helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from map_image_check.ad_remote import (
    _is_computer_online,
    _parse_ad_computer_records,
    check_computers_online,
    classify_ad_computer,
)


class AdRemoteTests(unittest.TestCase):
    def test_classify_ad_computer_windows_server(self) -> None:
        self.assertTrue(
            classify_ad_computer(operating_system="Windows Server 2019 Standard")
        )

    def test_classify_ad_computer_workstation(self) -> None:
        self.assertFalse(classify_ad_computer(operating_system="Windows 11 Pro"))

    def test_classify_ad_computer_domain_controller_flag(self) -> None:
        self.assertTrue(classify_ad_computer(operating_system=None, user_account_control=8192))

    def test_parse_ad_computer_records(self) -> None:
        records = _parse_ad_computer_records(
            [
                {"Name": "WS01", "OperatingSystem": "Windows 10 Pro", "IsServer": False},
                {"Name": "SRV01", "OperatingSystem": "Windows Server 2016", "IsServer": True},
            ]
        )
        self.assertEqual(len(records), 2)
        by_name = {record.name: record for record in records}
        self.assertFalse(by_name["WS01"].is_server)
        self.assertTrue(by_name["SRV01"].is_server)

    @patch("map_image_check.ad_remote._ping_host")
    @patch("map_image_check.ad_remote._tcp_port_open")
    def test_is_computer_online_prefers_smb(self, mock_tcp, mock_ping) -> None:
        mock_tcp.return_value = True
        self.assertTrue(_is_computer_online("buh-ws03"))
        mock_tcp.assert_called_once()
        mock_ping.assert_not_called()

    @patch("map_image_check.ad_remote._ping_host")
    @patch("map_image_check.ad_remote._tcp_port_open")
    def test_is_computer_online_falls_back_to_ping(self, mock_tcp, mock_ping) -> None:
        mock_tcp.return_value = False
        mock_ping.return_value = True
        self.assertTrue(_is_computer_online("buh-ws03"))
        mock_ping.assert_called_once()

    @patch("map_image_check.ad_remote._is_computer_online")
    def test_check_computers_online(self, mock_online) -> None:
        def side_effect(name: str, **_kwargs: object) -> bool:
            return name in {"PC-ON-1", "PC-ON-2"}

        mock_online.side_effect = side_effect
        result = check_computers_online(["PC-ON-2", "PC-OFF", "PC-ON-1"])
        self.assertTrue(result["PC-ON-1"])
        self.assertTrue(result["PC-ON-2"])
        self.assertFalse(result["PC-OFF"])

    @patch("map_image_check.ad_remote._is_computer_online", return_value=True)
    def test_check_computers_online_progress(self, _mock_online) -> None:
        seen: list[tuple[int, int, str | None]] = []

        def on_progress(done: int, total: int, name: str | None) -> None:
            seen.append((done, total, name))

        check_computers_online(["A", "B"], progress_callback=on_progress)
        self.assertEqual(seen, [(1, 2, "A"), (2, 2, "B")] or [(1, 2, "B"), (2, 2, "A")])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][1], 2)
        self.assertEqual(seen[1][0], 2)


if __name__ == "__main__":
    unittest.main()
