#!/usr/bin/env python3
"""
capacity_test.py
----------------
Tests local Ollama capacity on any single machine (Dell, Spark, or any node).
Reads model_registry from connectivity_config.json, computes theoretical
N_hw for each model at 8K context, then empirically validates by attempting
to load and run inference on each installed model.

Produces a capacity report showing:
  - Theoretical agent count (N_hw) per model
  - AFS (Agent Feasibility Score) vs FOO defaults
  - Empirical pass/fail per model
  - VRAM vs RAM spill annotation per model
  - Recommended MAX configuration for this machine

Usage:
    python capacity_test.py                     # test all installed models
    python capacity_test.py --quick             # skip inference, check status only
    python capacity_test.py --model mistral:7b  # test a single model
    python capacity_test.py --tier 3            # test only tier-3 models
    python capacity_test.py --host 172.24.0.1   # remote Ollama (e.g. WSL→Windows)

Notes:
  - ctx8k Modelfile variants (e.g. llama3.3:70b-ctx8k) must be created on
    EACH machine where they will be used. Run the Modelfile commands on
    this machine if T1 models show "ctx8k variant missing".
  - arm64_crash status applies only to ARM64 machines (NVIDIA Spark GB10).
    The same models work on x86_64 (Dell).
  - Cold load times (first inference after Ollama starts) are expected to
    be slow — especially for large models loading from RAM.
"""

import argparse
import json
import math
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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
        cfg = json.load(f)
    if "memory_budget" not in cfg:
        print("ERROR: 'memory_budget' section missing from connectivity_config.json.")
        print("Update your config from the latest connectivity_config.json.template.")
        sys.exit(1)
    return cfg


# ==============================================================================
# PLATFORM DETECTION
# ==============================================================================

def detect_architecture() -> str:
    """Return 'arm64' or 'x86_64'."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x86_64"


# ==============================================================================
# MEMORY BUDGET
# ==============================================================================

def n_hw(available_gb: float, total_agent_gb: float) -> int:
    if total_agent_gb <= 0:
        return 0
    return math.floor(available_gb / total_agent_gb)


def n_foo(p: float, confidence: float) -> int:
    if p <= 0 or p >= 1:
        return 999
    return math.ceil(math.log(1 - confidence) / math.log(p))


def afs(available_gb: float, total_agent_gb: float,
        p: float, confidence: float) -> float:
    hw  = n_hw(available_gb, total_agent_gb)
    foo = n_foo(p, confidence)
    return float("inf") if foo == 0 else hw / foo


def infer_timeout(total_8k_gb: float) -> float:
    """
    Scale inference timeout with model size.
    Small models cold-load in ~30s; large models may need 3+ minutes.
    """
    if total_8k_gb < 5:
        return 60.0
    elif total_8k_gb < 15:
        return 120.0
    elif total_8k_gb < 50:
        return 180.0
    else:
        return 300.0


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


def strip_thinking(text: str) -> str:
    """
    Remove thinking blocks from model output.
    Handles both <think>...</think> (DeepSeek-R1) and
    Ollama's 'Thinking...\n...\n...done thinking.\n' pattern (Qwen3).
    Returns the cleaned reply, or empty string if nothing remains.
    """
    import re
    # Remove <think>...</think> blocks (DeepSeek-R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove Ollama thinking header/footer (Qwen3 native format)
    text = re.sub(r"Thinking\.\.\..*?\.\.\.done thinking\.", "", text, flags=re.DOTALL)
    return text.strip()


def run_inference(base_url: str, model: str,
                  timeout: float = 60.0) -> tuple[bool, str, float]:
    """
    Returns (success, reply_or_error, elapsed_seconds).

    For thinking models (qwen3, deepseek-r1, nemotron) we do NOT suppress
    thinking mode — suppression via /no_think can produce empty replies when
    the model has nothing to say outside the think block. Instead we allow
    the full response and strip thinking tags from the output. max_tokens is
    raised to 500 to ensure the final answer is not truncated.
    """
    url      = f"{base_url}/v1/chat/completions"
    messages = [{"role": "user", "content": "Reply with exactly one word: ready"}]

    payload = json.dumps({
        "model"     : model,
        "messages"  : messages,
        "max_tokens": 500,   # generous — thinking blocks can be long
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
            raw     = body["choices"][0]["message"]["content"]
            reply   = strip_thinking(raw)
            if not reply:
                # Fall back to the raw content so we can see what came back
                reply_disp = raw.strip()[:80].replace("\n", " ") if raw.strip() else "(empty)"
                return False, f"empty after stripping think tags — raw: {reply_disp}", elapsed
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
# INSTALL DETECTION
# ==============================================================================

def check_installed(ollama_name: str, base_name: str,
                    installed_models: list[str]) -> tuple[bool, bool]:
    """
    Returns (base_installed, variant_installed).

    For models with a -ctx8k suffix (variants), we check separately
    whether the base and the derived variant are installed.
    This allows the script to distinguish:
      - variant installed → can test directly
      - base installed but variant missing → suggest Modelfile command
      - neither installed → not available
    """
    is_variant      = ollama_name != base_name
    # Exact match only — avoids gemma3:1b matching gemma3:4b etc.
    variant_installed = ollama_name in installed_models
    base_installed    = base_name   in installed_models
    return base_installed, variant_installed


# ==============================================================================
# RESULT DATACLASS
# ==============================================================================

@dataclass
class ModelResult:
    name             : str
    ollama_name      : str
    base_name        : str           # name without -ctx8k suffix
    tier             : Optional[int]
    total_8k_gb      : float
    n_hw_spark       : int
    n_hw_dell        : int
    afs_spark        : float
    afs_dell         : float
    base_installed   : bool
    variant_installed: bool
    tested           : bool
    passed           : bool
    reply            : str   = ""
    elapsed_s        : float = 0.0
    skip_reason      : str   = ""


# ==============================================================================
# MODELFILE COMMANDS
# ==============================================================================

def modelfile_command(base_name: str, ollama_name: str) -> str:
    """
    Return shell commands to create a ctx8k Modelfile variant.
    Emits PowerShell syntax on Windows, bash syntax on Linux/macOS.
    """
    safe = base_name.replace(":", "_").replace("/", "_")

    if platform.system() == "Windows":
        # PowerShell here-string: closing "@ MUST be flush-left (no indentation).
        # We return a single pre-formatted block rather than joining lines,
        # to guarantee no leading whitespace is added by the caller.
        path = f"$env:TEMP\\Modelfile.{safe}"
        return (
            f'@"\n' +
            f'FROM {base_name}\n' +
            f'PARAMETER num_ctx 8192\n' +
            f'"@ | Out-File -FilePath "{path}" -Encoding utf8\n' +
            f'ollama create {ollama_name} -f "{path}"'
        )
    else:
        # bash heredoc
        path  = f"/tmp/Modelfile.{safe}"
        lines = [
            f"cat > {path} << 'EOF'",
            f"FROM {base_name}",
            "PARAMETER num_ctx 8192",
            "EOF",
            f"ollama create {ollama_name} -f {path}",
        ]
        return "\n".join(lines)


# ==============================================================================
# REPORT
# ==============================================================================

def fmt_afs(v: float) -> str:
    return "  ∞  " if v == float("inf") else f"{v:.2f}"


def vram_label(total_gb: float, vram_gb: float) -> str:
    if total_gb <= vram_gb:
        return f"VRAM ({total_gb:.1f}/{vram_gb:.0f} GB)"
    spill = total_gb - vram_gb
    return f"VRAM+RAM (spill {spill:.1f} GB to RAM)"


def print_report(results: list[ModelResult],
                 foo_n: int, spark_gb: float,
                 dell_gb: float, dell_vram_gb: float,
                 arch: str):
    W = 84
    print()
    print("=" * W)
    print(" ALICE / MAX — Local Capacity Report")
    print(f" Architecture: {arch}  |  FOO N_min={foo_n}  |  "
          f"Spark: {spark_gb:.0f} GB unified  |  "
          f"Dell: {dell_vram_gb:.0f} GB VRAM + {dell_gb:.0f} GB RAM")
    print("=" * W)

    hdr = (f"  {'Model':<32} {'GB':>5} "
           f"{'N_hw(Sp)':>9} {'AFS(Sp)':>8} "
           f"{'N_hw(De)':>9} {'AFS(De)':>8}  Status")
    print(hdr)
    print("-" * W)

    missing_variants = []

    for r in sorted(results, key=lambda x: (x.tier or 99, x.total_8k_gb)):
        tier_str = f"T{r.tier}" if r.tier else " -"

        if r.skip_reason:
            status = f"SKIP ({r.skip_reason})"
        elif not r.base_installed and not r.variant_installed:
            status = "not installed"
        elif r.base_installed and not r.variant_installed and r.ollama_name != r.base_name:
            status = "base installed — ctx8k variant missing (see below)"
            missing_variants.append(r)
        elif not r.tested:
            install_note = "installed" if r.variant_installed else "base only"
            status = f"{install_note}, not tested (--quick)"
        elif r.passed:
            cold = "  [cold load]" if r.elapsed_s > 20 else ""
            status = f"✓  '{r.reply}'  ({r.elapsed_s:.1f}s){cold}"
        else:
            err = r.reply[:50] + "…" if len(r.reply) > 50 else r.reply
            status = f"✗  {err}"

        print(f"  [{tier_str}] {r.name:<30} {r.total_8k_gb:>4.1f}  "
              f"{r.n_hw_spark:>8}  {fmt_afs(r.afs_spark):>8}  "
              f"{r.n_hw_dell:>8}  {fmt_afs(r.afs_dell):>8}  {status}")

    print("-" * W)

    # ctx8k variant creation commands
    if missing_variants:
        print()
        print(" ctx8k VARIANTS MISSING ON THIS MACHINE")
        print(" Run these commands to create them:")
        print("-" * W)
        for r in missing_variants:
            print(f"  # {r.base_name}")
            # Indent each line of the command block for readability,
            # EXCEPT on Windows where the here-string closing "@ must
            # remain flush-left. We print it unindented in that case.
            cmd = modelfile_command(r.base_name, r.ollama_name)
            import platform as _pf
            if _pf.system() == "Windows":
                print(cmd)
            else:
                for cmd_line in cmd.split("\n"):
                    print(f"    {cmd_line}")
            print()

    # ── Recommended MAX config ──────────────────────────────────────────────────
    working = [r for r in results if r.passed and r.tier is not None]
    print()
    print(f" RECOMMENDED MAX CONFIG  (N_foo = {foo_n})")
    print("-" * W)

    if not working:
        print("  No working models confirmed yet.")
        print("  Run without --quick once models are installed and variants created.")
        print("=" * W)
        print()
        return

    # Sort ascending by size — smallest first favours VRAM fit
    by_size = sorted(working, key=lambda x: x.total_8k_gb)

    # ── Config A: VRAM-optimal (models that fit entirely in VRAM) ─────────────
    vram_pool   = dell_vram_gb
    vram_config = []
    for r in by_size:
        if r.total_8k_gb <= vram_pool:
            vram_config.append(r)
            vram_pool -= r.total_8k_gb
        if len(vram_config) >= foo_n:
            break

    vram_total = sum(r.total_8k_gb for r in vram_config)
    vram_afs   = min((r.afs_dell for r in vram_config), default=0.0)

    print(f"  CONFIG A — VRAM-only (fast inference, no RAM spill)")
    print(f"  {'─'*60}")
    if vram_config:
        for r in vram_config:
            print(f"    ✓ {r.ollama_name:<44} {r.total_8k_gb:>5.1f} GB  VRAM")
        print(f"    {'─'*55}")
        print(f"    Agents : {len(vram_config)} / {foo_n} needed")
        print(f"    VRAM   : {vram_total:.1f} / {dell_vram_gb:.0f} GB used")
        afs_note = "✓ feasible" if vram_afs >= 1.0 else f"⚠ AFS={vram_afs:.2f} — insufficient for full FOO round"
        print(f"    AFS    : {vram_afs:.2f}  {afs_note}")
    else:
        print("    No models fit simultaneously in VRAM.")
    print()

    # ── Config B: RAM-extended (fill remaining FOO slots from RAM) ────────────
    # Start from VRAM config, add next-smallest models that fit in remaining RAM
    ram_config  = list(vram_config)
    ram_used    = vram_total
    for r in by_size:
        if r in ram_config:
            continue
        if ram_used + r.total_8k_gb <= dell_gb and len(ram_config) < foo_n:
            ram_config.append(r)
            ram_used += r.total_8k_gb

    ram_total    = sum(r.total_8k_gb for r in ram_config)
    ram_spill    = max(0.0, ram_total - dell_vram_gb)
    ram_afs      = min((r.afs_dell for r in ram_config), default=0.0)

    print(f"  CONFIG B — RAM-extended (reaches N_foo={foo_n}, slower for spilled models)")
    print(f"  {'─'*60}")
    if ram_config:
        for r in ram_config:
            in_vram = r in vram_config
            loc     = "VRAM      " if in_vram else f"RAM spill "
            print(f"    {'✓' if in_vram else '⚠'} {r.ollama_name:<44} "
                  f"{r.total_8k_gb:>5.1f} GB  {loc}")
        print(f"    {'─'*55}")
        print(f"    Agents    : {len(ram_config)} / {foo_n} needed")
        print(f"    VRAM used : {min(ram_total, dell_vram_gb):.1f} / {dell_vram_gb:.0f} GB")
        if ram_spill > 0:
            print(f"    RAM spill : {ram_spill:.1f} GB  ⚠ spilled models run significantly slower")
        print(f"    RAM left  : {dell_gb - ram_total:.1f} GB")
        afs_note = "✓ feasible" if ram_afs >= 1.0 else f"⚠ AFS={ram_afs:.2f} — insufficient for full FOO round"
        print(f"    AFS (min) : {ram_afs:.2f}  {afs_note}")
    print()

    # ── Spark recommendation ──────────────────────────────────────────────────
    if ram_afs < 1.0 or len(ram_config) < foo_n:
        print(f"  NOTE: Full FOO round (N={foo_n}, AFS≥1.0) requires the Spark.")
        print(f"  The Dell is recommended for development and single-agent tasks.")
        spark_working = [r for r in working if r.n_hw_spark >= 1]
        spark_config  = []
        spark_pool    = spark_gb
        for r in sorted(spark_working, key=lambda x: x.total_8k_gb, reverse=True):
            if r.total_8k_gb <= spark_pool and len(spark_config) < foo_n:
                spark_config.append(r)
                spark_pool -= r.total_8k_gb
        if spark_config:
            spark_total = sum(r.total_8k_gb for r in spark_config)
            spark_afs   = min(r.afs_spark for r in spark_config)
            print()
            print(f"  SPARK CONFIG — full FOO round")
            print(f"  {'─'*60}")
            for r in spark_config:
                print(f"    ✓ {r.ollama_name:<44} {r.total_8k_gb:>5.1f} GB")
            print(f"    {'─'*55}")
            print(f"    Total  : {spark_total:.1f} / {spark_gb:.0f} GB")
            afs_note = "✓ feasible" if spark_afs >= 1.0 else f"⚠ AFS={spark_afs:.2f}"
            print(f"    AFS    : {spark_afs:.2f}  {afs_note}")

    print("=" * W)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ALICE local capacity test")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=11434)
    parser.add_argument("--model", default=None,
                        help="Test a single model by registry key or ollama_name")
    parser.add_argument("--tier",  type=int, default=None,
                        help="Test only models of this tier (1, 2, or 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip inference — check install status only")
    args = parser.parse_args()

    cfg          = load_config()
    mb           = cfg["memory_budget"]
    registry     = mb["model_registry"]
    foo_cfg      = mb["foo_consensus"]["default_assumption"]
    hw           = mb["hardware"]
    arch         = detect_architecture()

    spark_gb     = float(hw["spark"]["available_gb"])
    dell_gb      = float(hw["dell"]["available_gb"])
    dell_vram_gb = float(hw["dell"].get("vram_gb", 16))
    foo_p        = float(foo_cfg["p"])
    foo_c        = float(foo_cfg["C"])
    foo_n        = int(foo_cfg["N_foo"])

    base_url = get_base_url(args.host, args.port)

    print(f"\nDetected architecture : {arch}")
    print(f"Checking Ollama at    : {base_url} ...", end=" ", flush=True)
    if not check_ollama_running(base_url):
        print("OFFLINE")
        print("  On Windows : start Ollama from the system tray, or: ollama serve")
        print("  On Linux   : sudo systemctl start ollama")
        print("  From WSL   : python capacity_test.py --host <windows_gateway_ip>")
        sys.exit(1)
    print("OK")

    installed_models = list_installed_models(base_url)
    print(f"Installed models      : {installed_models}\n")

    results = []

    for key, spec in registry.items():
        # Skip _comment entries and any non-dict values
        if not isinstance(spec, dict):
            continue

        ollama_nm = spec["ollama_name"]
        base_nm   = key           # registry key is always the base name
        status    = spec.get("status", "untested")
        tier      = spec.get("tier")
        total_gb  = float(spec["total_8k_gb"])

        if args.model and args.model not in (key, ollama_nm):
            continue
        if args.tier and tier != args.tier:
            continue

        hw_spark = n_hw(spark_gb, total_gb)
        hw_dell  = n_hw(dell_gb,  total_gb)
        afs_s    = afs(spark_gb, total_gb, foo_p, foo_c)
        afs_d    = afs(dell_gb,  total_gb, foo_p, foo_c)

        # Platform-aware skip: arm64_crash only applies on ARM64
        if status == "arm64_crash" and arch == "arm64":
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, base_name=base_nm,
                tier=tier, total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                base_installed=False, variant_installed=False,
                tested=False, passed=False,
                skip_reason="arm64_crash"
            ))
            continue

        if status == "too_large":
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, base_name=base_nm,
                tier=tier, total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                base_installed=False, variant_installed=False,
                tested=False, passed=False,
                skip_reason="too_large"
            ))
            continue

        base_installed, variant_installed = check_installed(
            ollama_nm, base_nm, installed_models
        )

        # If variant is missing but base is present, report but don't test
        if not variant_installed and ollama_nm != base_nm and base_installed:
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, base_name=base_nm,
                tier=tier, total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                base_installed=True, variant_installed=False,
                tested=False, passed=False
            ))
            continue

        # Determine which name to actually call
        call_name = ollama_nm if variant_installed else (base_nm if base_installed else None)

        if args.quick or call_name is None:
            results.append(ModelResult(
                name=key, ollama_name=ollama_nm, base_name=base_nm,
                tier=tier, total_8k_gb=total_gb,
                n_hw_spark=hw_spark, n_hw_dell=hw_dell,
                afs_spark=afs_s, afs_dell=afs_d,
                base_installed=base_installed,
                variant_installed=variant_installed,
                tested=False, passed=False
            ))
            continue

        timeout = infer_timeout(total_gb)
        print(f"  Testing {call_name:<40} (timeout {timeout:.0f}s) ...",
              end=" ", flush=True)
        ok, reply, elapsed = run_inference(base_url, call_name, timeout=timeout)
        print(f"{'OK' if ok else 'FAIL'}  ({elapsed:.1f}s)")

        results.append(ModelResult(
            name=key, ollama_name=ollama_nm, base_name=base_nm,
            tier=tier, total_8k_gb=total_gb,
            n_hw_spark=hw_spark, n_hw_dell=hw_dell,
            afs_spark=afs_s, afs_dell=afs_d,
            base_installed=base_installed,
            variant_installed=variant_installed,
            tested=True, passed=ok,
            reply=reply, elapsed_s=elapsed
        ))

    print_report(results, foo_n, spark_gb, dell_gb, dell_vram_gb, arch)


if __name__ == "__main__":
    main()