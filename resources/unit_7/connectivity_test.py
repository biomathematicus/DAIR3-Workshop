#!/usr/bin/env python3
"""
connectivity_test.py
--------------------
Generic connectivity tester for LLM infrastructure.
Tests TCP reachability, Ollama API, and inference end-to-end.

This is an example in which a Dell workstation connects to
a NVIDIA Spark workstation running LLMs. It shows how to have
two computers talk to each other on a university network.

CONFIGURATION:
    Machine profiles and ports are loaded from 'connectivity_config.json',
    which must be created in the same directory as this script.

    connectivity_config.json is NOT committed to version control — it
    contains IP addresses and MAC addresses specific to your local
    environment. Make sure it is listed in .gitignore:

        # .gitignore entry:
        connectivity_config.json

    Use connectivity_config.json.template (safe to commit) as your
    starting point. Copy it, rename it, and fill in your values.

    Template structure:
    {
        "machines": {
            "spark": {
                "label"       : "NVIDIA Spark (LLM server)",
                "ip"          : "0.0.0.0",
                "mac"         : "00:00:00:00:00:00",
                "description" : "Spark GB10 running Ollama"
            },
            "dell": {
                "label"       : "Dell (dev workstation)",
                "ip"          : "0.0.0.0",
                "mac"         : "00:00:00:00:00:00",
                "description" : "Dell running software stack"
            },
            "local": {
                "label"       : "Localhost (single-machine mode)",
                "ip"          : "127.0.0.1",
                "mac"         : null,
                "description" : "Single-machine dev mode"
            }
        },
        "ports": {
            "ollama" : 11434,
            "vllm"   : 8000,
            "app"    : 8080
        },
        "test_defaults": {
            "model"              : "mistral:7b",
            "profile"            : "spark",
            "timeout_tcp"        : 3.0,
            "timeout_api"        : 5.0,
            "timeout_inference"  : 60.0
        }
    }

    MAC address format: either colons (aa:bb:cc:dd:ee:ff) or dashes
    (AA-BB-CC-DD-EE-FF) are accepted — the script normalizes both to
    lowercase colon format before comparison. Windows 'ipconfig /all'
    uses dashes; Linux 'ip a' and 'arp' use colons.

    Model names: use the -ctx8k Modelfile variants for large models to
    avoid runaway context window memory allocation on Ollama. Safe
    defaults: mistral:7b (always works), qwen3:32b-ctx8k,
    llama3.3:70b-ctx8k, nemotron-3-super:120b-ctx8k.

    Firewall note (informational — not enforced by this script):
    On the Spark, restrict Ollama to known client IPs:
        sudo ufw allow from <DELL_IP> to any port 11434
        sudo ufw deny 11434
    Optional MAC-layer restriction (iptables, same L2 segment only —
    not applicable when machines communicate through a router):
        sudo iptables -A INPUT -p tcp --dport 11434 \
            ! --mac-source <DELL_MAC> -j DROP

Usage:
    python connectivity_test.py
    python connectivity_test.py --profile spark
    python connectivity_test.py --profile local
    python connectivity_test.py --profile spark --skip-inference
    python connectivity_test.py --profile spark --model mistral:7b
"""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ==============================================================================
# CONFIG LOADER
# ==============================================================================

CONFIG_FILE = Path(__file__).parent / "connectivity_config.json"


def load_config() -> dict[str, Any]:
    """
    Load connectivity_config.json from the same directory as this script.
    Exits with a clear message if the file is missing or malformed.
    """
    if not CONFIG_FILE.exists():
        print(f"ERROR: Configuration file not found: {CONFIG_FILE}")
        print()
        print("Create connectivity_config.json in the same directory as this")
        print("script. Use connectivity_config.json.template as your starting")
        print("point — copy it, rename it, and fill in your IP/MAC values.")
        print()
        print("connectivity_config.json must be listed in .gitignore.")
        sys.exit(1)

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: connectivity_config.json is not valid JSON: {e}")
        sys.exit(1)


# Load once at module level so dataclass defaults can reference config values
_config            = load_config()
MACHINES           = _config["machines"]
PORTS              = _config["ports"]
_test_defaults     = _config["test_defaults"]

TIMEOUT_TCP_SECONDS       = float(_test_defaults["timeout_tcp"])
TIMEOUT_API_SECONDS       = float(_test_defaults["timeout_api"])
TIMEOUT_INFERENCE_SECONDS = float(_test_defaults["timeout_inference"])
DEFAULT_TEST_MODEL        = _test_defaults["model"]
DEFAULT_PROFILE           = _test_defaults["profile"]

# Models known to use chain-of-thought / thinking mode by default.
# For these, we inject a system prompt disabling thinking so the inference
# smoke test stays fast and within max_tokens budget.
THINKING_MODELS = ("qwen3", "deepseek-r1")

INFERENCE_MAX_TOKENS  = 50
INFERENCE_PROMPT      = "Reply with exactly one word: ready"
INFERENCE_NO_THINK_SP = "/no_think"   # Ollama system-prompt directive


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TestResult:
    name         : str
    passed       : bool
    detail       : str
    elapsed_s    : Optional[float] = None
    informational: bool = False   # if True, failure does not fail the suite


@dataclass
class TestConfig:
    target_ip   : str
    target_mac  : Optional[str]
    target_label: str
    ollama_port : int = PORTS["ollama"]
    vllm_port   : int = PORTS["vllm"]
    test_model  : str = DEFAULT_TEST_MODEL
    results     : list = field(default_factory=list)


# ==============================================================================
# MAC NORMALIZATION
# ==============================================================================

def normalize_mac(mac: str) -> str:
    """
    Normalize a MAC address to lowercase colon-separated format.

    Accepts both common formats:
      - Windows 'ipconfig /all' style: AC-91-A1-7D-06-FF  (dashes, upper)
      - Linux 'ip a' / 'arp' style:   ac:91:a1:7d:06:ff  (colons, lower)

    Returns lowercase colon format: ac:91:a1:7d:06:ff
    """
    # Replace dashes with colons, then lowercase
    return mac.replace("-", ":").lower()


# ==============================================================================
# INDIVIDUAL TESTS
# ==============================================================================

def test_tcp(cfg: TestConfig) -> TestResult:
    """Raw TCP handshake — confirms host is reachable and port is open."""
    label = f"TCP {cfg.target_ip}:{cfg.ollama_port}"
    t0 = time.time()
    try:
        with socket.create_connection(
            (cfg.target_ip, cfg.ollama_port),
            timeout=TIMEOUT_TCP_SECONDS
        ):
            elapsed = time.time() - t0
            return TestResult("tcp_connect", True, f"{label} reachable", elapsed)
    except socket.timeout:
        return TestResult("tcp_connect", False,
            f"{label} — timeout ({TIMEOUT_TCP_SECONDS}s). "
            "Host unreachable or firewall dropping packets.")
    except ConnectionRefusedError:
        return TestResult("tcp_connect", False,
            f"{label} — refused. Ollama not running or bound to wrong interface.")
    except OSError as e:
        return TestResult("tcp_connect", False, f"{label} — OS error: {e}")


def test_ollama_models(cfg: TestConfig) -> TestResult:
    """GET /v1/models — confirms Ollama is serving and returns model list."""
    url = f"http://{cfg.target_ip}:{cfg.ollama_port}/v1/models"
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=TIMEOUT_API_SECONDS) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read().decode())
            models = [m["id"] for m in body.get("data", [])]
            detail = (f"Models available: {models}"
                      if models else "No models pulled yet — run: ollama pull <model>")
            return TestResult("ollama_models", True, detail, elapsed)
    except urllib.error.URLError as e:
        return TestResult("ollama_models", False, f"HTTP error: {e.reason}")
    except Exception as e:
        return TestResult("ollama_models", False, f"Unexpected: {e}")


def _is_thinking_model(model_name: str) -> bool:
    """Return True if the model is known to use thinking/CoT mode by default."""
    return any(model_name.lower().startswith(prefix) for prefix in THINKING_MODELS)


def test_ollama_inference(cfg: TestConfig) -> TestResult:
    """
    POST /v1/chat/completions — end-to-end inference smoke test.

    For models with built-in chain-of-thought reasoning (e.g. qwen3),
    a system prompt directive disables thinking mode so the test stays
    fast and within max_tokens budget. Without this, Ollama returns
    HTTP 500 when max_tokens is exhausted mid-think.
    """
    url = f"http://{cfg.target_ip}:{cfg.ollama_port}/v1/chat/completions"

    messages = []
    if _is_thinking_model(cfg.test_model):
        messages.append({"role": "system", "content": INFERENCE_NO_THINK_SP})
    messages.append({"role": "user", "content": INFERENCE_PROMPT})

    payload = json.dumps({
        "model"     : cfg.test_model,
        "messages"  : messages,
        "max_tokens": INFERENCE_MAX_TOKENS,
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_INFERENCE_SECONDS) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read().decode())
            reply = body["choices"][0]["message"]["content"].strip()
            thinking_note = " [thinking disabled]" if _is_thinking_model(cfg.test_model) else ""
            return TestResult("ollama_inference", True,
                f"Model '{cfg.test_model}' replied: '{reply}'{thinking_note}", elapsed)
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes).get("error", body_bytes.decode())
        except Exception:
            detail = body_bytes.decode()
        return TestResult("ollama_inference", False, f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        return TestResult("ollama_inference", False, f"HTTP error: {e.reason}")
    except (KeyError, IndexError):
        return TestResult("ollama_inference", False, "Unexpected response format")
    except Exception as e:
        return TestResult("ollama_inference", False, f"Unexpected: {e}")


def test_mac_reachability(cfg: TestConfig) -> TestResult:
    """
    Informational MAC check — verifies ARP table has an entry for target IP.

    Only meaningful on the same L2 segment (direct switch, no router between
    machines). When machines communicate through a university or corporate
    network, traffic crosses a router and ARP entries are not visible between
    hosts — this is expected and correct behavior, not an error.

    MAC addresses are normalized to lowercase colon format before comparison
    so that Windows-style dashes (AA-BB-CC-DD-EE-FF) and Linux-style colons
    (aa:bb:cc:dd:ee:ff) are treated as equivalent.

    This test is always marked informational: a failure here does not
    indicate a connectivity problem and does not fail the suite.

    NOTE: MACs can be spoofed; IP firewall rules are the primary control.
    """
    if cfg.target_mac is None:
        return TestResult("mac_arp", True,
            "Skipped — MAC not configured (loopback or not applicable)",
            informational=True)

    import subprocess
    try:
        result = subprocess.run(
            ["arp", "-n", cfg.target_ip],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.lower()
        # Normalize expected MAC to colon/lowercase to match arp output format
        expected_mac = normalize_mac(cfg.target_mac)

        if expected_mac in output:
            return TestResult("mac_arp", True,
                f"ARP entry matches expected MAC {expected_mac}",
                informational=True)
        elif cfg.target_ip in output:
            lines = [l for l in output.splitlines() if cfg.target_ip in l]
            return TestResult("mac_arp", False,
                f"ARP entry found but MAC mismatch. "
                f"Expected: {expected_mac}. "
                f"Seen: {lines[0] if lines else 'unknown'}. "
                f"Possible spoofing or wrong constant.",
                informational=True)
        else:
            return TestResult("mac_arp", False,
                f"No ARP entry for {cfg.target_ip} — machines are on different "
                "L2 segments (routed network). This is expected and not an error.",
                informational=True)
    except FileNotFoundError:
        return TestResult("mac_arp", False,
            "'arp' command not found. Install net-tools or use 'ip neigh'",
            informational=True)
    except Exception as e:
        return TestResult("mac_arp", False, f"ARP check error: {e}",
            informational=True)


# ==============================================================================
# LOCAL INTERFACE INFO
# ==============================================================================

def show_local_info():
    hostname = socket.gethostname()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
    except Exception:
        primary_ip = "unavailable"
    print(f"  This machine : {hostname} ({primary_ip})")


# ==============================================================================
# REPORT
# ==============================================================================

def print_report(cfg: TestConfig) -> bool:
    width = 60
    print("=" * width)
    print(" LLM Infrastructure Connectivity Test")
    print("=" * width)
    print(f"  Target  : {cfg.target_label}")
    print(f"  IP      : {cfg.target_ip}")
    mac_display = normalize_mac(cfg.target_mac) if cfg.target_mac else "not configured"
    print(f"  MAC     : {mac_display}")
    print(f"  Model   : {cfg.test_model}")
    show_local_info()
    print("-" * width)

    real_results = [r for r in cfg.results if not r.informational]
    info_results = [r for r in cfg.results if r.informational]

    for r in cfg.results:
        icon   = "ℹ" if r.informational else ("✓" if r.passed else "✗")
        suffix = "  [informational]" if r.informational else ""
        timing = f"  ({r.elapsed_s:.2f}s)" if r.elapsed_s is not None else ""
        print(f"  {icon} {r.name:<22} {r.detail}{timing}{suffix}")

    print("-" * width)
    all_passed = all(r.passed for r in real_results)
    print(f"  {'✓ All tests passed' if all_passed else '✗ Some tests failed'}")
    if info_results:
        print(f"  (informational checks do not affect pass/fail)")
    print("=" * width)
    return all_passed


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM infrastructure connectivity test"
    )
    parser.add_argument(
        "--profile",
        choices=list(MACHINES.keys()),
        default=DEFAULT_PROFILE,
        help=f"Target machine profile (default: {DEFAULT_PROFILE})"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TEST_MODEL,
        help=f"Model for inference test (default: {DEFAULT_TEST_MODEL})"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference test (faster, no model load required)"
    )
    args = parser.parse_args()

    machine = MACHINES[args.profile]
    cfg = TestConfig(
        target_ip    = machine["ip"],
        target_mac   = machine.get("mac"),
        target_label = machine["label"],
        test_model   = args.model,
    )

    # Run tests in sequence — stop at first real failure.
    # MAC ARP is always run when TCP succeeds (informational, never a gate).
    tcp_result = test_tcp(cfg)
    cfg.results.append(tcp_result)

    if tcp_result.passed:
        mac_result = test_mac_reachability(cfg)
        cfg.results.append(mac_result)

        models_result = test_ollama_models(cfg)
        cfg.results.append(models_result)

        if models_result.passed and not args.skip_inference:
            inference_result = test_ollama_inference(cfg)
            cfg.results.append(inference_result)

    all_passed = print_report(cfg)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()