"""
Helpers for remote scanning via Active Directory and UNC shares.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

_POWERSHELL_EXE = "powershell"
_REMOTE_SHARE_CHOICES = tuple(f"{chr(letter)}$" for letter in range(ord("C"), ord("Z") + 1))
_DEFAULT_REMOTE_SHARES = ("C$",)
_ONLINE_CHECK_WORKERS = 24
_SMB_PORT = 445
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 4.0
_UAC_SERVER_TRUST_ACCOUNT = 0x2000


@dataclass(slots=True)
class AdComputerRecord:
    name: str
    is_server: bool
    operating_system: str | None = None


def remote_share_choices() -> tuple[str, ...]:
    return _REMOTE_SHARE_CHOICES


def default_remote_shares() -> tuple[str, ...]:
    return _DEFAULT_REMOTE_SHARES


def normalize_remote_share(share: str) -> str:
    value = share.strip().upper().rstrip("\\/")
    if not value:
        raise ValueError("Remote share name is empty.")
    if not value.endswith("$"):
        value += "$"
    if len(value) != 2 or not value[0].isalpha() or value[1] != "$":
        raise ValueError(f"Unsupported remote share name: {share!r}")
    return value


def build_unc_roots(computer_names: Iterable[str], shares: Iterable[str]) -> list[Path]:
    normalized_shares = [normalize_remote_share(share) for share in shares]
    if not normalized_shares:
        return []

    roots: list[Path] = []
    seen: set[str] = set()
    for computer_name in computer_names:
        host = str(computer_name).strip().strip("\\/ ")
        if not host:
            continue
        for share in normalized_shares:
            unc = f"\\\\{host}\\{share}\\"
            if unc.lower() in seen:
                continue
            seen.add(unc.lower())
            roots.append(Path(unc))
    return roots


def classify_ad_computer(
    *,
    operating_system: str | None,
    user_account_control: int | None = None,
) -> bool:
    """Return True if the AD computer object looks like a server (not a workstation)."""
    os_text = (operating_system or "").strip()
    if os_text and "server" in os_text.lower():
        return True
    if user_account_control is not None and (
        int(user_account_control) & _UAC_SERVER_TRUST_ACCOUNT
    ):
        return True
    return False


def _parse_ad_computer_records(data: object) -> list[AdComputerRecord]:
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError("Unexpected Active Directory output format.")

    records: list[AdComputerRecord] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            if name:
                records.append(AdComputerRecord(name=name, is_server=False))
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or item.get("name") or "").strip()
        if not name:
            continue
        os_value = item.get("OperatingSystem")
        if os_value is None:
            os_value = item.get("operatingSystem")
        operating_system = str(os_value).strip() if os_value else None
        if "IsServer" in item or "isServer" in item:
            is_server = bool(item.get("IsServer") if "IsServer" in item else item.get("isServer"))
        else:
            uac_raw = item.get("userAccountControl")
            uac = int(uac_raw) if uac_raw is not None else None
            is_server = classify_ad_computer(
                operating_system=operating_system,
                user_account_control=uac,
            )
        records.append(
            AdComputerRecord(
                name=name,
                is_server=is_server,
                operating_system=operating_system or None,
            )
        )

    unique: dict[str, AdComputerRecord] = {}
    for record in records:
        unique[record.name] = record
    return sorted(unique.values(), key=lambda record: record.name.lower())


def list_enabled_ad_computer_records(
    *,
    powershell_exe: str = _POWERSHELL_EXE,
    timeout_seconds: int = 60,
) -> list[AdComputerRecord]:
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
function ConvertTo-AdComputerRecord {
    param(
        [string]$Name,
        [string]$OperatingSystem,
        [int]$UserAccountControl = 0
    )
    $os = [string]$OperatingSystem
    $isServer = ($os -match 'Server') -or (($UserAccountControl -band 8192) -ne 0)
    [PSCustomObject]@{
        Name = $Name
        OperatingSystem = $os
        IsServer = [bool]$isServer
    }
}
$module = Get-Module -ListAvailable -Name ActiveDirectory
if ($module) {
    Import-Module ActiveDirectory
    $items = Get-ADComputer -Filter 'Enabled -eq $true' -Properties Name, OperatingSystem, userAccountControl |
        ForEach-Object {
            ConvertTo-AdComputerRecord -Name $_.Name -OperatingSystem $_.OperatingSystem -UserAccountControl $_.userAccountControl
        } |
        Sort-Object Name
}
else {
    Add-Type -AssemblyName System.DirectoryServices
    $root = [ADSI]'LDAP://RootDSE'
    $defaultNamingContext = [string]$root.defaultNamingContext
    if (-not $defaultNamingContext) {
        throw 'Could not determine the default naming context for the current domain.'
    }

    $searchRoot = New-Object System.DirectoryServices.DirectoryEntry("LDAP://$defaultNamingContext")
    $searcher = New-Object System.DirectoryServices.DirectorySearcher($searchRoot)
    $searcher.PageSize = 1000
    $searcher.SearchScope = [System.DirectoryServices.SearchScope]::Subtree
    $searcher.Filter = '(&(objectCategory=computer)(!(userAccountControl:1.2.840.113556.1.4.803:=2)))'
    [void]$searcher.PropertiesToLoad.Add('name')
    [void]$searcher.PropertiesToLoad.Add('operatingSystem')
    [void]$searcher.PropertiesToLoad.Add('userAccountControl')

    $results = $searcher.FindAll()
    $items = foreach ($result in $results) {
        if (-not $result.Properties.Contains('name')) { continue }
        $name = [string]$result.Properties['name'][0]
        $os = ''
        if ($result.Properties.Contains('operatingSystem')) {
            $os = [string]$result.Properties['operatingSystem'][0]
        }
        $uac = 0
        if ($result.Properties.Contains('userAccountControl')) {
            $uac = [int]$result.Properties['userAccountControl'][0]
        }
        ConvertTo-AdComputerRecord -Name $name -OperatingSystem $os -UserAccountControl $uac
    }
    $items = $items | Sort-Object Name
}
$items | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            [
                powershell_exe,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while querying Active Directory computers after {timeout_seconds} seconds."
        ) from exc

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if not message:
            message = "PowerShell returned a non-zero exit code."
        raise RuntimeError(f"Failed to query Active Directory computers. {message}")

    raw = result.stdout.strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse PowerShell Active Directory output: {raw[:300]}"
        ) from exc

    return _parse_ad_computer_records(data)


def list_enabled_ad_computers(
    *,
    powershell_exe: str = _POWERSHELL_EXE,
    timeout_seconds: int = 60,
) -> list[str]:
    return [record.name for record in list_enabled_ad_computer_records(
        powershell_exe=powershell_exe,
        timeout_seconds=timeout_seconds,
    )]


def _normalize_computer_name(computer_name: str) -> str:
    return str(computer_name).strip().strip("\\/ ")


def _tcp_port_open(
    host: str,
    port: int,
    *,
    timeout_seconds: float,
) -> bool:
    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, socket.gaierror):
        return False

    for family, socktype, proto, _canonname, sockaddr in addresses:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(timeout_seconds)
                sock.connect(sockaddr)
                return True
        except OSError:
            continue
    return False


def _ping_host(host: str, *, timeout_seconds: float) -> bool:
    if sys.platform == "win32":
        wait_ms = max(500, int(timeout_seconds * 1000))
        try:
            result = subprocess.run(
                ["ping", "-n", "1", "-w", str(wait_ms), host],
                capture_output=True,
                timeout=timeout_seconds + 2.0,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0

    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(max(1, int(timeout_seconds))), host],
            capture_output=True,
            timeout=timeout_seconds + 2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _is_computer_online(
    computer_name: str,
    *,
    timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> bool:
    """
    Consider a host online if SMB (445) or ICMP ping succeeds.

    SMB is checked first because remote scanning uses UNC shares and many
    domain PCs block ping while still serving file sharing.
    """
    host = _normalize_computer_name(computer_name)
    if not host:
        return False
    if _tcp_port_open(host, _SMB_PORT, timeout_seconds=timeout_seconds):
        return True
    return _ping_host(host, timeout_seconds=timeout_seconds)


def check_computers_online(
    computer_names: Iterable[str],
    *,
    max_workers: int = _ONLINE_CHECK_WORKERS,
    per_host_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, bool]:
    """Return True for hosts reachable via SMB (445) or ping."""
    names = sorted({_normalize_computer_name(name) for name in computer_names if _normalize_computer_name(name)})
    if not names:
        return {}

    total = len(names)
    completed = 0
    workers = max(1, min(max_workers, len(names)))
    online: dict[str, bool] = {name: False for name in names}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _is_computer_online,
                name,
                timeout_seconds=per_host_timeout_seconds,
            ): name
            for name in names
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                online[name] = bool(future.result())
            except Exception:
                online[name] = False
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, name)
    return online
