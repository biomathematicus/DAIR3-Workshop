#!/usr/bin/env python3
"""
capacity_test.py
----------------
Tests local Ollama capacity on any single machine (Dell, Spark, or any node).
Reads model_registry from connectivity_config.json, computes theoretical
N_hw for each model, then empirically validates by attempting to load and
run inference on each model that fits.

Produces a capacity report showing:
  - Theoretical agent count (N_hw) per model at 8K context
  - AFS (Agent Feasibility Score) vs FOO defaults
  - Empirical pass/fail for each model
  - Recommended MAX configuration for this machine

Usage:
    python capacity_test.py                     # test all installed models
    python capacity_test.py --quick             # skip inference, check install only
    python capacity_test.py --model mistral:7b  # test a single model
    python capacity_test.py --tier 3            # test only tier-3 models
    python capacity_test.py --host 172.24.0.1   # target remote Ollama (e.g. WSL→Windows)
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG_FILE = Path(__file__).parent / "connectivity_config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"ERROR: {CONFIG_FILE} not found.")
        print("Copy connectivity_config.json.template → connectivity_config.json")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


# ==============================================================================
# MEMORY BUDGET
# ==============================================================================

def n_hw(available_gb: float, total_agent_gb: float) -> int:
    """Maximum simultaneous agents given hardware budget."""
    if total_agent_gb <= 0:
        return 0
    return math.floor(available_gb / total_agent_gb)


def n_foo(p: float, confidence: float) -> int:
    """Minimum agents for FOO Byzantine consensus at given confidence."""
    if p <= 0 or p >= 1:
        return 999
    return math.ceil(math.log(1 - confidence) / math.log(p))


def afs(available_gb: float, total_agent_gb: float,
        p: float, confidence: float) -> float:
    """Agent Feasibility Score = N_hw / N_foo. Must be >= 1.0."""
    hw  = n_hw(available_gb, total_agent_gb)
    foo = n_foo(p, confidence)
    return float("inf") if foo == 0 else hw / foo


# ==============================================================================
# OLLAMA API HELPERS
# ==============================================================================

def get_base_url(host: str = "127.0.0.1", port: int = 11434) -> str:
    return f"http://{host}:{port}"


def check_ollama_running(base_url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/v1/models"),
            timeout=timeout
        ):
            return True
    except Exception:
        return False


def list_installed_models(base_url: str) -> list[str]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/v1/models"),
            timeout=5
        ) as resp:
            body = json.loads(resp.read().decode())
            return [m["id"] for m in body.get("data", [])]
    except Exception:
        return []


THINKING_PREFIXES = ("qwen3", "deepseek-r1", "nemotron")


def is_thinking_model(name: str) -> bool:
    return any(name.lower().startswith(p) for p in THINKING_PREFIXES)


def run_inference(base_url: str, model: str,
                  timeout: float = 60.0) -> tuple[bool, str, float]:
    """Returns (success, reply_or_error, elapsed_seconds)."""
    url      = f"{base_url}/v1/chat/completions"
    messages = []
    if is_thinking_model(model):
        messages.append({"role": "system", "content": "/no_think"})
    messages.append({"role": "user",
                     "content": "Reply with exactly one word: ready"})

    payload = json.dumps({
        "model"     : model,
        "messages"  : messages,
        "max_tokens": 50,
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.time() - t0
            body    = json.loads(resp.read().decode())
            reply   = body["choices"][0]["message"]["content"].strip()
            return True, reply, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        try:
            detail = json.loads(e.read()).get("error", str(e))
        except Exception:
            detail = str(e)
        return False, f"HTTP {e.code}: {detail}", elapsed
    except Exception as e:
        return False, str(e), time.time() - t0


# ==============================================================================
# RESULT DATACLASS
# ==============================================================================

@dataclass
class ModelResult:
    name        : str
    ollama_name : str
    tier        : Optional[int]
    total_8k_gb : float
    n_hw_spark  : int
    n_hw_dell   : int
    afs_spark   : float
    afs_dell    : float
    installed   : bool
    tested      : bool
    passed      : bool
    reply       : str   = ""
    elapsed_s   : float = 0.0
    skip_reason : str   = ""


# ==============================================================================
# REPORT
# ==============================================================================

def fmt_afs(v: float) -> str:
    return "  ∞ " if v == float("inf") else f"{v:.2f}"


def print_report(results: list[ModelResult],
                 foo_n: int, spark_gb: float, dell_gb: float,
                 dell_vram_gb: float):
    W = 82
    print()
    print("=" * W)
    print(" ALICE / MAX — Local Capacity Report")
    print(f" FOO N_min={foo_n}  |  "
          f"Spark: {spark_gb:.0f} GB unified  |  "
          f"Dell: {dell_gb:.0f} GB RAM + {dell_vram_gb:.0f} GB VRAM")
    print("=" * W)

    hdr = (f"  {'Model':<32} {'GB':>5} "
           f"{'N_hw(Sp)':>9} {'AFS(Sp)':>8} "
           f"{'N_hw(De)':>9} {'AFS(De)':>8}  Status")
    print(hdr)
    print("-" * W)

    for r in sorted(results, key=lambda x: (x.tier or 99, x.total_8k_gb)):
        tier_str = f"T{r.tier}" if r.tier else " -"

        if r.skip_reason:
            status = f"SKIP ({r.skip_reason})"
        elif not r.installed:
            status = "not installed"
        elif not r.tested:
            status = "installed, not tested (--quick)"
        elif r.passed:
            status = f"✓  '{r.reply}'  ({r.elapsed_s:.1f}s)"
        else:
            # Truncate long error messages
            err = r.reply[:45] + "…" if len(r.reply) > 45 else r.reply
            status = f"✗  {err}"

        print(f"  [{tier_str}] {r.name:<30} {r.total_8k_gb:>4.1f}  "
              f"{r.n_hw_spark:>8}  {fmt_afs(r.afs_spark):>8}  "
              f"{r.n_hw_dell:>8}  {fmt_afs(r.afs_dell):>8}  {status}")

    print("-" * W)

    # Recommended MAX config for THIS machine
    working = [r for r in results if r.passed and r.tier is not None]
    print()
    print(f" RECOMMENDED MAX CONFIG — this machine  (N_foo = {foo_n})")
    print("-" * W)
    if not working:
        print("  No working models confirmed yet.")
        print("  Run without --quick once models are installed.")
    else:
        total = 0.0
        slots = sorted(working, key=lambda x: x.total_8k_gb, reverse=True)[:5]
        for r in slots:
            total    += r.total_8k_gb
            feasible  = "✓" if r.afs_dell >= 1.0 else "⚠"
            vram_note = ("  ← fits in VRAM"
                         if r.total_8k_gb <= dell_vram_gb else
                         "  ← spills to RAM")
            print(f"  {feasible} {r.ollama_name:<42} "
                  f"{r.total_8k_gb:>5.1f} GB{vram_note}")
        print(f"  {'─' * 60}")
        print(f"  Subtotal : {total:.1f} GB")
        print(f"  Headroom : {dell_gb - total:.1f} GB  "
              f"(VRAM headroom: {dell_vram_gb - min(total, dell_vram_gb):.1f} GB)")

    print("=" * W)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ALICE local capacity test")
    parser.add_argument("--host",  default="127.0.0.1",
                        help="Ollama host (default: 127.0.0.1)")
    parser.add_argument("--port",  type=int, default=11434,
                        help="Ollama port (default: 11434)")
    parser.add_argument("--model", default=None,
                        help="Test a single model by registry key or ollama_name")
    parser.add_argument("--tier",  type=int, default=None,
                        help="Test only models of this tier (1, 2, or 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip inference — check install status only")
    args = parser.parse_args()

    cfg      = load_config()
    mb       = cfg.get("memory_budget", {})

    if not mb:
        print("ERROR: 'memory_budget' section missing from connectivity_config.json.")
        print("Update your config from the latest connectivity_config.json.template.")
        sys.exit(1)

    registry     = mb["model_registry"]
    foo_cfg      = mb["foo_consensus"]["default_assumption"]
    hw           = mb["hardware"]

    spark_gb     = float(hw["spark"]["available_gb"])
    dell_gb      = float(hw["dell"]["available_gb"])
    # RTX A5500 has 16 GB VRAM — update dell.vram_gb in config if different
    dell_vram_gb = float(hw["dell"].get("vram_gb", 16))
    foo_p        = float(foo_cfg["p"])
    foo_c        = float(foo_cfg["C"])
    foo_n        = int(foo_cfg["N_foo"])

    base_url = get_base_url(args.host, args.port)

    print(f"\nChecking Ollama at {base_url} ...", end=" ", flush=True)
    if not check_ollama_running(base_url):
        print("OFFLINE")
        print(f"  Ollama is not running at {base_url}.")
        print("  On Windows : start Ollama from the system tray, or: ollama serve")
        print("  On Linux   : sudo systemctl start ollama")
        print("  From WSL   : python capacity_test.py --host <windows_gateway_ip>")
        sys.exit(1)
    print("OK")

    installed_models = list_installed_models(base_url)
    print(f"  Installed: {installed_models}\n")

    results = []

    for key, spec in registry.items():
        # Skip the _comment entry and any non-dict values
        if not isinstance(spec, dict):
            continue

        ollama_nm = spec["ollama_name"]
        status    = spec.get("status", "untested")
        tier      = spec.get("tier")
        total_gb  = float(spec["total_8k_gb"])

        # Filter flags
        if args.model and args.model not in (key, ollama_nm):
            continue
        if args.tier and tier != args.tier:
            continue

        hw_spark = n_hw(spark_gb, total_gb)
        hw_dell  = n_hw(dell_gb,  total_gb)
        afs_s    = afs(spark_gb, total_gb, foo_p, foo_c)
        afs_d    = afs(dell_gb,  total_gb, foo_p, foo_c)

        # Hard skips
        if status in ("too_large", "arm64_crash"):
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, tier=tier,
                total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                installed=False, tested=False, passed=False,
                skip_reason=status
            ))
            continue

        # Check if installed (match on full name or base name)
        installed = any(
            ollama_nm == m or key.split(":")[0] in m
            for m in installed_models
        )

        if args.quick or not installed:
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, tier=tier,
                total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                installed=installed, tested=False, passed=False
            ))
            continue

        # Run inference
        print(f"  Testing {ollama_nm} ...", end=" ", flush=True)
        ok, reply, elapsed = run_inference(
            base_url, ollama_nm,
            timeout=float(cfg["test_defaults"]["timeout_inference"])
        )
        print(f"{'OK' if ok else 'FAIL'}  ({elapsed:.1f}s)")

        results.append(ModelResult(
            name=key, ollama_name=ollama_nm, tier=tier,
            total_8k_gb=total_gb,
            n_hw_spark=hw_spark, n_hw_dell=hw_dell,
            afs_spark=afs_s, afs_dell=afs_d,
            installed=installed, tested=True,
            passed=ok, reply=reply, elapsed_s=elapsed
        ))

    print_report(results, foo_n, spark_gb, dell_gb, dell_vram_gb)


if __name__ == "__main__":
    main()