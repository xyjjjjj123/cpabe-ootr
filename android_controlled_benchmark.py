import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT, pair


# ============================================================
# Settings
# ============================================================

PAIRING_GROUP = "SS512"
L_VALUES = [10, 30, 50]

WARMUP = 10
ITERATIONS = 50

# Longer than the previous script because this is the final thermal-controlled run.
BLOCK_COOLDOWN_SEC = 20.0
PAIR_GAP_SEC = 0.10
ITERATION_GAP_SEC = 0.15

group = PairingGroup(PAIRING_GROUP)


# ============================================================
# Device / thermal helpers
# ============================================================

def safe_shell(command):
    try:
        r = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def read_float_file(path):
    """Read a numeric sysfs value without guessing its unit."""
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def read_battery_temp_c():
    candidates = [
        "/sys/class/power_supply/battery/temp",
        "/sys/class/power_supply/bms/temp",
    ]
    for p in candidates:
        v = read_float_file(p)
        if v is not None:
            # Battery temp often uses deci-Celsius (e.g., 350 -> 35C).
            # Some devices expose milli-Celsius; handle both conventions.
            if abs(v) >= 1000:
                v /= 1000.0
            elif abs(v) > 80:
                v /= 10.0
            return v if -20 <= v <= 100 else None
    return None


def read_max_thermal_zone_c():
    temps = []
    base = Path("/sys/class/thermal")
    try:
        for p in base.glob("thermal_zone*/temp"):
            v = read_float_file(p)
            if v is not None:
                # Thermal-zone temp is normally in milli-Celsius.
                if abs(v) >= 1000:
                    v /= 1000.0
                elif abs(v) > 150:
                    v /= 10.0
                if -20 <= v <= 150:
                    temps.append(v)
    except Exception:
        pass
    return max(temps) if temps else None


def thermal_snapshot():
    return {
        "battery_temp_c": read_battery_temp_c(),
        "max_thermal_zone_c": read_max_thermal_zone_c(),
    }


def android_metadata():
    def getprop(name):
        return safe_shell(["getprop", name])

    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "android_model": getprop("ro.product.model"),
        "android_device": getprop("ro.product.device"),
        "android_manufacturer": getprop("ro.product.manufacturer"),
        "android_version": getprop("ro.build.version.release"),
        "android_sdk": getprop("ro.build.version.sdk"),
        "android_build_fingerprint": getprop("ro.build.fingerprint"),
        "kernel": safe_shell(["uname", "-a"]),
        "initial_thermal": thermal_snapshot(),
    }


def cooldown(label):
    print(
        f"  Cooldown before {label}: {BLOCK_COOLDOWN_SEC:.0f}s",
        flush=True,
    )
    time.sleep(BLOCK_COOLDOWN_SEC)


# ============================================================
# Statistics
# ============================================================

def mean_std(values):
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def pct_reduction(baseline, ours):
    if baseline == 0:
        return float("nan")
    return 100.0 * (baseline - ours) / baseline


# ============================================================
# Cryptographic helpers
# ============================================================

def assemble_reencrypted_ct(policy_ref, c0_prime, c1_hat, c2_hat, c1_prime, c2_prime):
    l = len(c1_hat)
    assert len(c2_hat) == l
    assert len(c1_prime) == l
    assert len(c2_prime) == l

    rows = [
        (c1_hat[i], c2_hat[i], c1_prime[i], c2_prime[i])
        for i in range(l)
    ]
    return (policy_ref, c0_prime, rows)


def sample_nonzero_z():
    """Sample from Z_p^*, matching the valid recovery domain."""
    zero = group.init(ZR, 0)
    z = group.random(ZR)
    while z == zero:
        z = group.random(ZR)
    return z


def prepare_token(c1, c2):
    z = sample_nonzero_z()
    c1_hat = [c ** z for c in c1]
    c2_hat = [c ** z for c in c2]

    return {
        "z": z,
        "c1_hat": c1_hat,
        "c2_hat": c2_hat,
        "used": False,
    }


def consume_token(token):
    # Single-threaded in-memory state update; no concurrency guarantee.
    if token["used"]:
        raise RuntimeError("Token reuse detected.")
    token["used"] = True
    return token


def take_token(ctx, pools):
    """Lookup/consume one ciphertext-bound token inside the timed operation."""
    queue = pools[ctx["ct_id"]]
    if not queue:
        raise RuntimeError("Token pool exhausted; benchmark needs a fresh pool.")
    token = consume_token(queue[0])
    queue.popleft()
    return token


def test_phase(
    ask_list,
    h_base,
    attr_values,
    c1_prime,
    c2_prime,
    omegas,
    c0_prime,
):
    aggregate = None

    for ask, s_attr, c1p, c2p, omega in zip(
        ask_list, attr_values, c1_prime, c2_prime, omegas
    ):
        h_to_s = h_base ** s_attr

        n_i = (
            pair(ask, c1p)
            * pair(h_to_s, c2p)
        ) ** omega

        aggregate = n_i if aggregate is None else aggregate * n_i

    # Cost-only equality operation: the synthetic inputs are not a valid
    # encrypted policy instance. This comparison does NOT certify access.
    _ = (aggregate == c0_prime)
    return aggregate


def partial_decryption(
    ask_list,
    h_base,
    attr_values,
    c1_hat,
    c2_hat,
    omegas,
):
    aggregate = None

    for ask, s_attr, c1h, c2h, omega in zip(
        ask_list, attr_values, c1_hat, c2_hat, omegas
    ):
        h_to_s = h_base ** s_attr

        n_hat_i = (
            pair(ask, c1h)
            * pair(h_to_s, c2h)
        ) ** omega

        aggregate = (
            n_hat_i if aggregate is None else aggregate * n_hat_i
        )

    return aggregate


def final_unblinding(c0, aggregate_partial, z):
    one = group.init(ZR, 1)
    inv_z = one / z
    unblinded = aggregate_partial ** inv_z
    recovered_ck_like = c0 / unblinded
    _ = recovered_ck_like


# ============================================================
# Timed operations
# ============================================================

def baseline_online(ctx):
    # Fresh z is sampled in the online critical path.
    z = sample_nonzero_z()

    c1_hat = [c ** z for c in ctx["c1"]]
    c2_hat = [c ** z for c in ctx["c2"]]

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        c1_hat,
        c2_hat,
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime


def offline_token_operation(ctx):
    token = prepare_token(ctx["c1"], ctx["c2"])
    _ = token


def ours_online(ctx, pools):
    token = take_token(ctx, pools)

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        token["c1_hat"],
        token["c2_hat"],
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime


def baseline_receiver(ctx):
    z = sample_nonzero_z()

    c1_hat = [c ** z for c in ctx["c1"]]
    c2_hat = [c ** z for c in ctx["c2"]]

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        c1_hat,
        c2_hat,
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime

    final_unblinding(
        ctx["c0"],
        ctx["aggregate_partial"],
        z,
    )


def ours_receiver(ctx, pools):
    token = take_token(ctx, pools)

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        token["c1_hat"],
        token["c2_hat"],
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime

    final_unblinding(
        ctx["c0"],
        ctx["aggregate_partial"],
        token["z"],
    )


def baseline_path(ctx):
    z = sample_nonzero_z()

    c1_hat = [c ** z for c in ctx["c1"]]
    c2_hat = [c ** z for c in ctx["c2"]]

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        c1_hat,
        c2_hat,
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime

    test_phase(
        ctx["ask_list"],
        ctx["h_base"],
        ctx["attr_values"],
        ctx["c1_prime"],
        ctx["c2_prime"],
        ctx["omegas"],
        ctx["c0_prime"],
    )

    aggregate_partial = partial_decryption(
        ctx["ask_list"],
        ctx["h_base"],
        ctx["attr_values"],
        c1_hat,
        c2_hat,
        ctx["omegas"],
    )

    final_unblinding(
        ctx["c0"],
        aggregate_partial,
        z,
    )


def ours_path(ctx, pools):
    token = take_token(ctx, pools)

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        token["c1_hat"],
        token["c2_hat"],
        ctx["c1_prime"],
        ctx["c2_prime"],
    )
    _ = ct_prime

    test_phase(
        ctx["ask_list"],
        ctx["h_base"],
        ctx["attr_values"],
        ctx["c1_prime"],
        ctx["c2_prime"],
        ctx["omegas"],
        ctx["c0_prime"],
    )

    aggregate_partial = partial_decryption(
        ctx["ask_list"],
        ctx["h_base"],
        ctx["attr_values"],
        token["c1_hat"],
        token["c2_hat"],
        ctx["omegas"],
    )

    final_unblinding(
        ctx["c0"],
        aggregate_partial,
        token["z"],
    )


# ============================================================
# Input generation and timing
# ============================================================

def make_context(l):
    return {
        "l": l,
        # Reference to (M, rho); generated outside all timed regions.
        "policy_ref": ("M", "rho", l),
        # Unique ID for this synthetic ciphertext; outside online timing.
        "ct_id": os.urandom(16).hex(),
        "c1": [group.random(G1) for _ in range(l)],
        "c2": [group.random(G1) for _ in range(l)],
        "c1_prime": [group.random(G1) for _ in range(l)],
        "c2_prime": [group.random(G1) for _ in range(l)],
        "c0_prime": group.random(GT),
        "ask_list": [group.random(G1) for _ in range(l)],
        "h_base": group.random(G1),
        "attr_values": [group.random(ZR) for _ in range(l)],
        "omegas": [group.random(ZR) for _ in range(l)],
        "c0": group.random(GT),
        # Synthetic input for the recovery-operation microbenchmark only.
        # It is not a valid TA/AA partial-decryption result.
        "aggregate_partial": group.random(GT),
    }


def build_token_pool(ctx, count):
    """Pre-build a single-ciphertext FIFO pool outside online timing."""
    return {
        ctx["ct_id"]: deque(
            prepare_token(ctx["c1"], ctx["c2"]) for _ in range(count)
        )
    }


def timed_call(function, *args):
    gc.collect()
    was_enabled = gc.isenabled()

    before = thermal_snapshot()

    if was_enabled:
        gc.disable()

    try:
        start = time.perf_counter_ns()
        function(*args)
        end = time.perf_counter_ns()
    finally:
        if was_enabled:
            gc.enable()

    after = thermal_snapshot()

    return {
        "latency_ms": (end - start) / 1_000_000.0,
        "battery_temp_before_c": before["battery_temp_c"],
        "battery_temp_after_c": after["battery_temp_c"],
        "thermal_before_c": before["max_thermal_zone_c"],
        "thermal_after_c": after["max_thermal_zone_c"],
    }


# ============================================================
# Measurement blocks
# ============================================================

def measure_offline_vs_baseline(ctx, raw_rows):
    """
    Interleave:
        Baseline online re-encryption
        Ours offline token generation

    This is specifically designed to test whether the two 2l-E1 workloads
    remain comparable under the same thermal state.
    """
    baseline_samples = []
    offline_samples = []

    total = WARMUP + ITERATIONS

    for rep in range(total):
        measured = rep >= WARMUP
        measured_rep = rep - WARMUP + 1

        order = ["baseline", "offline"] if rep % 2 == 0 else ["offline", "baseline"]
        results = {}

        for name in order:
            if name == "baseline":
                results[name] = timed_call(baseline_online, ctx)
            else:
                results[name] = timed_call(offline_token_operation, ctx)

            if PAIR_GAP_SEC > 0:
                time.sleep(PAIR_GAP_SEC)

        if measured:
            baseline_samples.append(results["baseline"]["latency_ms"])
            offline_samples.append(results["offline"]["latency_ms"])

            for name in ("baseline", "offline"):
                r = results[name]
                raw_rows.append({
                    "l": ctx["l"],
                    "metric": "offline_vs_baseline",
                    "scheme": name,
                    "repetition": measured_rep,
                    "latency_ms": r["latency_ms"],
                    "order": "B-F" if rep % 2 == 0 else "F-B",
                    "battery_temp_before_c": r["battery_temp_before_c"],
                    "battery_temp_after_c": r["battery_temp_after_c"],
                    "thermal_before_c": r["thermal_before_c"],
                    "thermal_after_c": r["thermal_after_c"],
                })

        if ITERATION_GAP_SEC > 0:
            time.sleep(ITERATION_GAP_SEC)

        if measured and measured_rep % 10 == 0:
            print(
                f"    offline-vs-baseline: {measured_rep}/{ITERATIONS}",
                flush=True,
            )

    return offline_samples, baseline_samples


def paired_metric(
    ctx,
    baseline_function,
    ours_function,
    pools,
    metric_name,
    raw_rows,
):
    baseline_samples = []
    ours_samples = []

    total = WARMUP + ITERATIONS
    assert len(pools[ctx["ct_id"]]) == total

    for rep in range(total):
        measured = rep >= WARMUP
        measured_rep = rep - WARMUP + 1

        order = ["baseline", "ours"] if rep % 2 == 0 else ["ours", "baseline"]
        results = {}

        for name in order:
            if name == "baseline":
                results[name] = timed_call(baseline_function, ctx)
            else:
                results[name] = timed_call(ours_function, ctx, pools)

            if PAIR_GAP_SEC > 0:
                time.sleep(PAIR_GAP_SEC)

        if measured:
            baseline_samples.append(results["baseline"]["latency_ms"])
            ours_samples.append(results["ours"]["latency_ms"])

            for name in ("baseline", "ours"):
                r = results[name]
                raw_rows.append({
                    "l": ctx["l"],
                    "metric": metric_name,
                    "scheme": name,
                    "repetition": measured_rep,
                    "latency_ms": r["latency_ms"],
                    "order": "B-O" if rep % 2 == 0 else "O-B",
                    "battery_temp_before_c": r["battery_temp_before_c"],
                    "battery_temp_after_c": r["battery_temp_after_c"],
                    "thermal_before_c": r["thermal_before_c"],
                    "thermal_after_c": r["thermal_after_c"],
                })

        if ITERATION_GAP_SEC > 0:
            time.sleep(ITERATION_GAP_SEC)

        if measured and measured_rep % 10 == 0:
            print(
                f"    {metric_name}: {measured_rep}/{ITERATIONS}",
                flush=True,
            )

    assert not pools[ctx["ct_id"]], "Expected exactly one consumed token per run."
    return baseline_samples, ours_samples


# ============================================================
# Output
# ============================================================

# Keep original outputs intact, as timing now includes pool retrieval.
RAW_FILE = "android_thermal_token_pool_raw.csv"
SUMMARY_FILE = "android_thermal_token_pool_summary.csv"
META_FILE = "android_thermal_token_pool_metadata.json"


def save_raw(rows):
    fields = [
        "l",
        "metric",
        "scheme",
        "repetition",
        "latency_ms",
        "order",
        "battery_temp_before_c",
        "battery_temp_after_c",
        "thermal_before_c",
        "thermal_after_c",
    ]

    with open(RAW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows):
    fields = [
        "l",
        "offline_mean_ms",
        "offline_std_ms",
        "baseline_online_matched_mean_ms",
        "baseline_online_matched_std_ms",
        "offline_vs_baseline_ratio",
        "baseline_online_mean_ms",
        "baseline_online_std_ms",
        "ours_online_mean_ms",
        "ours_online_std_ms",
        "online_reduction_pct",
        "baseline_receiver_mean_ms",
        "baseline_receiver_std_ms",
        "ours_receiver_mean_ms",
        "ours_receiver_std_ms",
        "receiver_reduction_pct",
        "baseline_path_mean_ms",
        "baseline_path_std_ms",
        "ours_path_mean_ms",
        "ours_path_std_ms",
        "path_reduction_pct",
    ]

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata():
    metadata = {
        "device": android_metadata(),
        "benchmark": {
            "pairing_group": PAIRING_GROUP,
            "l_values": L_VALUES,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "all_android_samples_retained": True,
            "block_cooldown_sec": BLOCK_COOLDOWN_SEC,
            "pair_gap_sec": PAIR_GAP_SEC,
            "iteration_gap_sec": ITERATION_GAP_SEC,
            "candidate_set_assumption": (
                "|I| = l; one preselected candidate set; synthetic random inputs "
                "are not guaranteed to satisfy the policy"
            ),
            "input_validity": (
                "synthetic group elements and scalars, not an end-to-end "
                "valid ciphertext or verified successful access"
            ),
            "offline_baseline_order": "alternating B-F / F-B",
            "paired_metric_order": "alternating B-O / O-B",
            "scope": (
                "post-authentication operation-level timing with in-memory "
                "ciphertext-specific FIFO token lookup, single-threaded "
                "state check/update and assembly; offline preparation excluded "
                "from online timing; not application-level end-to-end latency"
            ),
            "token_pool": (
                "one ciphertext-specific deque per metric; one token per "
                "warm-up/measured invocation; no concurrency or persistence"
            ),
            "tokens_per_metric_pool": WARMUP + ITERATIONS,
            "note_on_storage_table": (
                "the separate serialized-storage experiment uses q=5; "
                "its size is not the RAM footprint of this benchmark's pool"
            ),
            "excluded": [
                "authentication",
                "network/IPC",
                "serialization",
                "persistent token storage and crash recovery",
                "concurrent synchronization",
                "candidate-set enumeration",
                "LSSS omega solving",
                "XOR",
                "AES payload decryption",
            ],
        },
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def run():
    print("# Thermal-controlled Android benchmark: timed in-memory FIFO lookup", flush=True)
    print("# Device/Platform:", platform.platform(), flush=True)
    print("# Python:", platform.python_version(), flush=True)
    print("# Pairing group:", PAIRING_GROUP, flush=True)
    print("# l values:", L_VALUES, flush=True)
    print("# warm-up:", WARMUP, flush=True)
    print("# measurements:", ITERATIONS, flush=True)
    print("# samples retained: all (no trimming)", flush=True)

    save_metadata()

    raw_rows = []
    summary_rows = []
    total = WARMUP + ITERATIONS

    for l in L_VALUES:
        print(f"\n========== l = {l} ==========", flush=True)

        ctx = make_context(l)

        # Pools are generated BEFORE their online measurement blocks.
        # Their generation is not included in Ours online timing.
        # Independent pools avoid sharing tokens between three metric blocks.
        # Each has 60 tokens for 10 warm-ups + 50 measured calls, not q=5.
        online_pools = build_token_pool(ctx, total)
        receiver_pools = build_token_pool(ctx, total)
        path_pools = build_token_pool(ctx, total)

        cooldown("offline/baseline matched block")
        print("  Measuring matched offline token vs baseline-online cost...", flush=True)
        offline_s, matched_baseline_s = measure_offline_vs_baseline(
            ctx,
            raw_rows,
        )

        cooldown("online re-encryption block")
        print("  Measuring Baseline vs Ours online re-encryption...", flush=True)
        base_online_s, ours_online_s = paired_metric(
            ctx,
            baseline_online,
            ours_online,
            online_pools,
            "online_reenc",
            raw_rows,
        )

        cooldown("receiver-side block")
        print("  Measuring receiver-side online cryptographic latency...", flush=True)
        base_receiver_s, ours_receiver_s = paired_metric(
            ctx,
            baseline_receiver,
            ours_receiver,
            receiver_pools,
            "receiver_online_crypto",
            raw_rows,
        )

        cooldown("decryption-path block")
        print("  Measuring online decryption-path cryptographic latency...", flush=True)
        base_path_s, ours_path_s = paired_metric(
            ctx,
            baseline_path,
            ours_path,
            path_pools,
            "decryption_path_online_crypto",
            raw_rows,
        )

        save_raw(raw_rows)

        off_m, off_sd = mean_std(offline_s)
        mb_m, mb_sd = mean_std(matched_baseline_s)

        bo_m, bo_sd = mean_std(base_online_s)
        oo_m, oo_sd = mean_std(ours_online_s)

        br_m, br_sd = mean_std(base_receiver_s)
        or_m, or_sd = mean_std(ours_receiver_s)

        bp_m, bp_sd = mean_std(base_path_s)
        op_m, op_sd = mean_std(ours_path_s)

        summary = {
            "l": l,
            "offline_mean_ms": off_m,
            "offline_std_ms": off_sd,
            "baseline_online_matched_mean_ms": mb_m,
            "baseline_online_matched_std_ms": mb_sd,
            "offline_vs_baseline_ratio": off_m / mb_m if mb_m else float("nan"),

            "baseline_online_mean_ms": bo_m,
            "baseline_online_std_ms": bo_sd,
            "ours_online_mean_ms": oo_m,
            "ours_online_std_ms": oo_sd,
            "online_reduction_pct": pct_reduction(bo_m, oo_m),

            "baseline_receiver_mean_ms": br_m,
            "baseline_receiver_std_ms": br_sd,
            "ours_receiver_mean_ms": or_m,
            "ours_receiver_std_ms": or_sd,
            "receiver_reduction_pct": pct_reduction(br_m, or_m),

            "baseline_path_mean_ms": bp_m,
            "baseline_path_std_ms": bp_sd,
            "ours_path_mean_ms": op_m,
            "ours_path_std_ms": op_sd,
            "path_reduction_pct": pct_reduction(bp_m, op_m),
        }

        summary_rows.append(summary)
        save_summary(summary_rows)

        print("\n  Summary for l =", l, flush=True)
        print(
            f"    Offline vs matched baseline: "
            f"{off_m:.3f} vs {mb_m:.3f} ms "
            f"(ratio={summary['offline_vs_baseline_ratio']:.3f})",
            flush=True,
        )
        print(
            f"    Online re-enc:   {bo_m:.3f} -> {oo_m:.3f} ms "
            f"({summary['online_reduction_pct']:.2f}% reduction)",
            flush=True,
        )
        print(
            f"    Receiver-side:   {br_m:.3f} -> {or_m:.3f} ms "
            f"({summary['receiver_reduction_pct']:.2f}% reduction)",
            flush=True,
        )
        print(
            f"    Decryption path: {bp_m:.3f} -> {op_m:.3f} ms "
            f"({summary['path_reduction_pct']:.2f}% reduction)",
            flush=True,
        )

    print("\n========== Experiment completed ==========", flush=True)
    print("Saved:", flush=True)
    print(" ", RAW_FILE, flush=True)
    print(" ", SUMMARY_FILE, flush=True)
    print(" ", META_FILE, flush=True)


if __name__ == "__main__":
    run()
