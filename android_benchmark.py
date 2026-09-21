import csv
import gc
import os
import platform
import statistics
import time
from collections import deque

# Operation-level benchmark with synthetic ciphertext components; in-memory
# FIFO lookup/check/update is timed. This is NOT a full protocol execution,
# concurrent/persistent token store, or verified successful authorization.
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT


PAIRING_GROUP = "SS512"

L_VALUES = [
    2, 4, 6, 8, 10,
    15, 20, 30, 40, 50
]

WARMUP = 10
ITERATIONS = 50

PRE_BLOCK_COOLDOWN_SEC = 3.0

# New output names preserve the original experiment's CSV files.
RAW_FILE = "android_fig3_token_pool_raw.csv"
SUMMARY_FILE = "android_fig3_token_pool_summary.csv"

group = PairingGroup(PAIRING_GROUP)


# ============================================================
# Ciphertext assembly
# ============================================================

def assemble_reencrypted_ct(
    policy_ref,
    c0_prime,
    c1_hat,
    c2_hat,
    c1_prime,
    c2_prime
):
    l = len(c1_hat)

    assert len(c2_hat) == l
    assert len(c1_prime) == l
    assert len(c2_prime) == l

    rows = [
        (
            c1_hat[i],
            c2_hat[i],
            c1_prime[i],
            c2_prime[i]
        )
        for i in range(l)
    ]

    return (
        policy_ref,
        c0_prime,
        rows
    )


# ============================================================
# Token preparation / consumption
# ============================================================

def sample_nonzero_z():
    """Use the nonzero blinding-factor domain needed by final recovery."""
    zero = group.init(ZR, 0)
    z = group.random(ZR)
    while z == zero:
        z = group.random(ZR)
    return z


def prepare_token(ctx):

    z = sample_nonzero_z()

    c1_hat = [
        c ** z
        for c in ctx["c1"]
    ]

    c2_hat = [
        c ** z
        for c in ctx["c2"]
    ]

    return {
        "z": z,
        "c1_hat": c1_hat,
        "c2_hat": c2_hat,
        "used": False,
    }


def consume_token(token):

    # Single-thread, in-memory state check/update only.
    # This does NOT implement concurrent atomicity or crash-safe persistence.

    if token["used"]:
        raise RuntimeError(
            "Token reuse detected."
        )

    token["used"] = True

    return token


def take_token(ctx, pools):
    """Timed ciphertext-specific FIFO lookup, check/update, and removal.

    Pool construction is outside the online timing. This is single-threaded
    in-memory consumption, not a concurrent or persistent storage mechanism.
    """
    queue = pools[ctx["ct_id"]]
    if not queue:
        raise RuntimeError("Token pool exhausted; benchmark needs a fresh pool.")
    token = consume_token(queue[0])
    queue.popleft()
    return token


# ============================================================
# Baseline online one-time re-encryption
#
# Fresh z
# + 2l exponentiations in G
# + ciphertext assembly
# ============================================================

def baseline_online(ctx):

    # Fresh z is generated in the request-time path.
    z = sample_nonzero_z()

    c1_hat = [
        c ** z
        for c in ctx["c1"]
    ]

    c2_hat = [
        c ** z
        for c in ctx["c2"]
    ]

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        c1_hat,
        c2_hat,
        ctx["c1_prime"],
        ctx["c2_prime"]
    )

    _ = ct_prime


# ============================================================
# Ours offline token preparation
#
# Fresh z_j
# + 2l exponentiations in G
# + token materialization
# ============================================================

def offline_token_operation(ctx):

    token = prepare_token(ctx)

    _ = token


# ============================================================
# Ours online token consumption
#
# 0 group exponentiations
# + ciphertext-specific in-memory FIFO lookup
# + single-use state transition
# + ciphertext assembly
# ============================================================

def ours_online(
    ctx,
    pools
):

    # Lookup AND consumption are inside timed_call's start/end boundaries.
    token = take_token(ctx, pools)

    ct_prime = assemble_reencrypted_ct(
        ctx["policy_ref"],
        ctx["c0_prime"],
        token["c1_hat"],
        token["c2_hat"],
        ctx["c1_prime"],
        ctx["c2_prime"]
    )

    _ = ct_prime


# ============================================================
# Input generation
#
# Existing ciphertext / policy material is generated outside
# timed regions.
# ============================================================

def make_context(l):

    return {
        "l": l,
        # Unique for this synthetic ciphertext; assigned outside timing.
        # The random group elements below are NOT from full CP-ABE encryption.
        "ct_id": os.urandom(16).hex(),

        # Existing (M, rho) policy metadata.
        # Its generation and LSSS processing are outside the
        # timed online re-encryption path.
        "policy_ref":
            ("M", "rho", l),

        "c1": [
            group.random(G1)
            for _ in range(l)
        ],

        "c2": [
            group.random(G1)
            for _ in range(l)
        ],

        "c1_prime": [
            group.random(G1)
            for _ in range(l)
        ],

        "c2_prime": [
            group.random(G1)
            for _ in range(l)
        ],

        "c0_prime":
            group.random(GT),
    }


# ============================================================
# Timing
# ============================================================

def timed_call(
    function,
    *args
):

    gc.collect()

    was_enabled = gc.isenabled()

    if was_enabled:
        gc.disable()

    try:

        start = time.perf_counter_ns()

        function(*args)

        end = time.perf_counter_ns()

    finally:

        if was_enabled:
            gc.enable()

    return (
        end - start
    ) / 1_000_000.0


def mean_std(values):

    return (
        statistics.mean(values),

        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )


# ============================================================
# Main experiment
# ============================================================

def run():

    print(
        "# Android Fig. 3: in-memory FIFO pool lookup + single-use update + assembly",
        flush=True
    )

    print(
        "# Device/Platform:",
        platform.platform(),
        flush=True
    )

    print(
        "# Python:",
        platform.python_version(),
        flush=True
    )

    print(
        "# Pairing group:",
        PAIRING_GROUP,
        flush=True
    )

    print(
        "# PID:",
        os.getpid(),
        flush=True
    )

    print(
        "# l values:",
        L_VALUES,
        flush=True
    )

    print(
        "# warm-up:",
        WARMUP,
        flush=True
    )

    print(
        "# measurements:",
        ITERATIONS,
        flush=True
    )

    raw_rows = []
    summary_rows = []

    total = (
        WARMUP
        + ITERATIONS
    )

    for l in L_VALUES:

        print(
            f"\n===== l = {l} =====",
            flush=True
        )

        ctx = make_context(l)

        # ----------------------------------------------------
        # Prepare independent single-use tokens for the
        # Ours-online measurements.
        #
        # Their preparation is deliberately outside the
        # Ours-online timing.
        #
        # Offline preparation latency is measured separately.
        # ----------------------------------------------------

        # A ciphertext-specific FIFO pool; its generation is NOT timed.
        # 60 tokens cover 10 warm-ups + 50 samples (not the q=5 storage test).
        online_pools = {
            ctx["ct_id"]: deque(prepare_token(ctx) for _ in range(total))
        }

        # ----------------------------------------------------
        # Small cooldown before each policy-size block.
        #
        # The separate thermal-controlled experiment provides
        # the stricter thermal/order-controlled validation.
        # ----------------------------------------------------

        if PRE_BLOCK_COOLDOWN_SEC > 0:
            time.sleep(
                PRE_BLOCK_COOLDOWN_SEC
            )

        baseline_samples = []
        offline_samples = []
        ours_samples = []

        # ----------------------------------------------------
        # Six execution permutations.
        #
        # This avoids always measuring Baseline, Offline and
        # Ours in the same execution order.
        # ----------------------------------------------------

        orders = (
            (
                "baseline",
                "offline",
                "ours"
            ),

            (
                "baseline",
                "ours",
                "offline"
            ),

            (
                "offline",
                "baseline",
                "ours"
            ),

            (
                "offline",
                "ours",
                "baseline"
            ),

            (
                "ours",
                "baseline",
                "offline"
            ),

            (
                "ours",
                "offline",
                "baseline"
            ),
        )

        for rep in range(total):

            measured = (
                rep >= WARMUP
            )

            measured_rep = (
                rep
                - WARMUP
                + 1
            )

            order = (
                orders[
                    rep
                    % len(orders)
                ]
            )

            result = {}

            for name in order:

                if name == "baseline":

                    result[name] = (
                        timed_call(
                            baseline_online,
                            ctx
                        )
                    )

                elif name == "offline":

                    result[name] = (
                        timed_call(
                            offline_token_operation,
                            ctx
                        )
                    )

                else:

                    result[name] = (
                        timed_call(
                            ours_online,
                            ctx,
                            online_pools
                        )
                    )

            # ------------------------------------------------
            # Warm-up samples are executed but not retained.
            # ------------------------------------------------

            if measured:

                baseline_samples.append(
                    result["baseline"]
                )

                offline_samples.append(
                    result["offline"]
                )

                ours_samples.append(
                    result["ours"]
                )

                raw_rows.extend([
                    {
                        "l":
                            l,

                        "metric":
                            "baseline_online",

                        "repetition":
                            measured_rep,

                        "latency_ms":
                            result["baseline"],

                        "order":
                            "-".join(order),
                    },

                    {
                        "l":
                            l,

                        "metric":
                            "ours_offline_token",

                        "repetition":
                            measured_rep,

                        "latency_ms":
                            result["offline"],

                        "order":
                            "-".join(order),
                    },

                    {
                        "l":
                            l,

                        "metric":
                            "ours_online",

                        "repetition":
                            measured_rep,

                        "latency_ms":
                            result["ours"],

                        "order":
                            "-".join(order),
                    },
                ])

        # ----------------------------------------------------
        # Statistics
        #
        # All Android samples are retained.
        # No trimming is performed.
        # ----------------------------------------------------

        bm, bs = mean_std(
            baseline_samples
        )

        fm, fs = mean_std(
            offline_samples
        )

        om, osd = mean_std(
            ours_samples
        )

        summary_rows.append({

            "l":
                l,

            "baseline_online_mean_ms":
                bm,

            "baseline_online_std_ms":
                bs,

            "ours_offline_mean_ms":
                fm,

            "ours_offline_std_ms":
                fs,

            "ours_online_mean_ms":
                om,

            "ours_online_std_ms":
                osd,
        })

        print(
            f"Baseline={bm:.6f}±{bs:.6f} ms, "
            f"Offline={fm:.6f}±{fs:.6f} ms, "
            f"Ours-online={om:.6f}±{osd:.6f} ms",
            flush=True,
        )

        assert not online_pools[ctx["ct_id"]], "Expected exactly one consumed token per run."

        # ----------------------------------------------------
        # Save checkpoint after each l.
        # ----------------------------------------------------

        with open(
            RAW_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "l",
                    "metric",
                    "repetition",
                    "latency_ms",
                    "order"
                ]
            )

            writer.writeheader()

            writer.writerows(
                raw_rows
            )

        with open(
            SUMMARY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            fields = [
                "l",
                "baseline_online_mean_ms",
                "baseline_online_std_ms",
                "ours_offline_mean_ms",
                "ours_offline_std_ms",
                "ours_online_mean_ms",
                "ours_online_std_ms",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    print(
        "\n# Experiment completed.",
        flush=True
    )

    print(
        "# Saved:",
        RAW_FILE,
        SUMMARY_FILE,
        flush=True
    )


if __name__ == "__main__":
    run()
