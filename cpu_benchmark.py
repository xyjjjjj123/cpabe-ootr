import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import gc
import os
import platform
import hashlib
from pathlib import Path

from charm.toolbox.pairinggroup import (
    PairingGroup, ZR, G1, GT, pair
)

group = PairingGroup('SS512')


def sample_nonzero_z():
    """Sample a nonzero one-time blinding factor for invertible recovery."""
    zero = group.init(ZR, 0)
    z = group.random(ZR)
    while z == zero:
        z = group.random(ZR)
    return z


# ============================================================
# Experiment settings
# ============================================================

ATTRIBUTES = [2, 4, 6, 8, 10, 15, 20]

WARMUP = 10
ITERATIONS = 50


# ============================================================
# Statistics
# ============================================================

def trimmed_mean_std(values, trim_ratio=0.1):

    values = np.sort(
        np.asarray(values, dtype=float)
    )

    n = len(values)
    k = int(n * trim_ratio)

    if n > 2 * k:
        values = values[k:n-k]

    mean = float(
        np.mean(values)
    )

    std = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )

    return mean, std


def save_partial_raw(raw_results, path="latency_raw_results_partial.csv"):
    """Save all completed l settings. Called only outside timed regions."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "l",
            "measurement",
            "offline_ours_ms",
            "online_base_ms",
            "online_ours_ms",
            "receiver_base_ms",
            "receiver_ours_ms",
            "path_base_ms",
            "path_ours_ms",
        ])

        for l in sorted(raw_results):
            samples = raw_results[l]
            for i in range(len(samples["offline_ours"])):
                writer.writerow([
                    l,
                    i + 1,
                    samples["offline_ours"][i],
                    samples["online_base"][i],
                    samples["online_ours"][i],
                    samples["receiver_base"][i],
                    samples["receiver_ours"][i],
                    samples["path_base"][i],
                    samples["path_ours"][i],
                ])


# ============================================================
# Batch sizes and execution-order control
#
# IMPORTANT: within each comparison family, Baseline and Ours use
# the SAME FIXED batch size for every l.  This keeps the reported
# per-request means and error bars on the same statistical footing.
#
# The values below make each timed interval long enough to suppress
# timer/scheduler noise without making the expensive path benchmark
# unnecessarily long.
# ============================================================

REENC_BATCH = 20
RECEIVER_BATCH = 8
PATH_BATCH = 2

# Counterbalancing removes the fixed "baseline first, ours later"
# block-order bias.  The re-encryption family has three metrics, so
# all six permutations are cycled.  Paired families alternate B/O.
REENC_ORDERS = (
    # The first two orders intentionally start with different metrics.
    # With 50 measured batches (8 full six-order cycles + 2 extras),
    # this keeps first/second/third-position counts as balanced as possible.
    ("online_base", "offline_ours", "online_ours"),
    ("offline_ours", "online_ours", "online_base"),
    ("online_ours", "online_base", "offline_ours"),
    ("online_base", "online_ours", "offline_ours"),
    ("online_ours", "offline_ours", "online_base"),
    ("offline_ours", "online_base", "online_ours"),
)

PAIR_ORDERS = (
    ("baseline", "ours"),
    ("ours", "baseline"),
)

ORDER_LOG_FILE = "latency_execution_order.csv"
METADATA_FILE = "latency_metadata.txt"


# ============================================================
# Core cryptographic operations
# ============================================================

def reencrypt(bases, z):

    return [
        c ** z
        for c in bases
    ]


def assemble_reencrypted_ct(c0_prime, c1_hat, c2_hat, c1_prime, c2_prime):
    """Assemble CT' in the same row-wise form used by the Android benchmark."""
    l = len(c1_hat)
    assert len(c2_hat) == l
    assert len(c1_prime) == l
    assert len(c2_prime) == l

    rows = [
        (c1_hat[i], c2_hat[i], c1_prime[i], c2_prime[i])
        for i in range(l)
    ]
    return (c0_prime, rows)


def prepare_token(c1, c2):
    """Prepare one ciphertext-bound token, including fresh z_j and state."""
    z = sample_nonzero_z()
    c1_hat = [c ** z for c in c1]
    c2_hat = [c ** z for c in c2]
    return {
        "z": z,
        "c1_hat": c1_hat,
        "c2_hat": c2_hat,
        "used": False,
    }


def test_phase(
    ask_list,
    h_base,
    attr_values,
    c1_prime,
    c2_prime,
    omegas,
    c0_prime
):
    """
    Benchmark the baseline TA/AA test-phase operation sequence for one
    already-selected candidate set I.  This mirrors the baseline paper:

        N_i = ( e(ASK_i, C'_{1,i}) * e(h^{s_i}, C'_{2,i}) )^{omega_i}.

    Candidate-set enumeration and LSSS coefficient solving are intentionally
    outside the timed region, as stated in the paper.
    """
    aggregate = None

    for ask, s_attr, c1p, c2p, omega in zip(
        ask_list,
        attr_values,
        c1_prime,
        c2_prime,
        omegas
    ):
        # One E1 per selected attribute, matching the baseline operation count.
        h_to_s = h_base ** s_attr

        n_i = (
            pair(ask, c1p)
            * pair(h_to_s, c2p)
        ) ** omega

        aggregate = (
            n_i
            if aggregate is None
            else aggregate * n_i
        )

    # Account for the final equality-test operation, as in the Android code.
    # Synthetic inputs are used here; this benchmark measures the operation
    # sequence rather than protocol-level soundness/correctness.
    _ = (aggregate == c0_prime)

    return aggregate


def partial_decryption(
    ask_list,
    h_base,
    attr_values,
    c1_hat,
    c2_hat,
    omegas
):
    """
    Benchmark the baseline TA/AA partial-decryption operation sequence:

        Nhat_i = ( e(ASK_i, Chat_{1,i}) * e(h^{s_i}, Chat_{2,i}) )^{omega_i}.

    The E1, E2, pairing, and multiplication counts therefore match the
    baseline paper and the audited Android benchmark.
    """
    aggregate = None

    for ask, s_attr, c1h, c2h, omega in zip(
        ask_list,
        attr_values,
        c1_hat,
        c2_hat,
        omegas
    ):
        # One E1 per selected attribute, matching the baseline operation count.
        h_to_s = h_base ** s_attr

        n_hat_i = (
            pair(ask, c1h)
            * pair(h_to_s, c2h)
        ) ** omega

        aggregate = (
            n_hat_i
            if aggregate is None
            else aggregate * n_hat_i
        )

    return aggregate


def final_unblinding_key_recovery(
    c0,
    aggregate_partial,
    z
):

    one = group.init(ZR, 1)

    inv_z = one / z

    unblinded = (
        aggregate_partial ** inv_z
    )

    _ = (
        c0 / unblinded
    )


# ============================================================
# Online one-time re-encryption
# ============================================================

def baseline_online_request(request):
    # Baseline paper: fresh one-time z is sampled after the access request.
    z = sample_nonzero_z()
    c1_hat = [c ** z for c in request["c1"]]
    c2_hat = [c ** z for c in request["c2"]]

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        c1_hat,
        c2_hat,
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime


def ours_online_request(request):
    # Single-thread operation-level model of the consume check-and-set.
    # Production protected storage must enforce this transition atomically.
    if request["used"]:
        raise RuntimeError("Token reuse detected.")
    request["used"] = True

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        request["c1_hat"],
        request["c2_hat"],
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime


# ============================================================
# Receiver-side online cryptographic path
#
# T_recv_online = T_online_reenc + T_recover
# ============================================================

def baseline_receiver_request(request):
    z = sample_nonzero_z()
    c1_hat = [c ** z for c in request["c1"]]
    c2_hat = [c ** z for c in request["c2"]]

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        c1_hat,
        c2_hat,
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime

    final_unblinding_key_recovery(
        request["c0"],
        request["aggregate_partial"],
        z
    )


def ours_receiver_request(request):
    # The token has already been selected/prepared outside the timed region.
    if request["used"]:
        raise RuntimeError("Token reuse detected.")
    request["used"] = True

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        request["c1_hat"],
        request["c2_hat"],
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime

    final_unblinding_key_recovery(
        request["c0"],
        request["aggregate_partial"],
        request["z"]
    )


# ============================================================
# Online decryption-path cryptographic path
#
# T_path_online =
# online re-encryption
# + TA/AA test
# + TA/AA partial decryption
# + receiver final unblinding/key recovery
# ============================================================

def baseline_path_request(request):
    # Receiver-side online one-time re-encryption
    z = sample_nonzero_z()
    c1_hat = [c ** z for c in request["c1"]]
    c2_hat = [c ** z for c in request["c2"]]

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        c1_hat,
        c2_hat,
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime

    # TA/AA test
    test_phase(
        request["ask_list"],
        request["h_base"],
        request["attr_values"],
        request["c1_prime"],
        request["c2_prime"],
        request["omegas"],
        request["c0_prime"]
    )

    # TA/AA partial decryption
    aggregate_partial = partial_decryption(
        request["ask_list"],
        request["h_base"],
        request["attr_values"],
        c1_hat,
        c2_hat,
        request["omegas"]
    )

    # Receiver final unblinding/key recovery
    final_unblinding_key_recovery(
        request["c0"],
        aggregate_partial,
        z
    )


def ours_path_request(request):
    # Online token consumption; token-pool lookup itself is not benchmarked.
    if request["used"]:
        raise RuntimeError("Token reuse detected.")
    request["used"] = True

    ct_prime = assemble_reencrypted_ct(
        request["c0_prime"],
        request["c1_hat"],
        request["c2_hat"],
        request["c1_prime"],
        request["c2_prime"],
    )
    _ = ct_prime

    # TA/AA test
    test_phase(
        request["ask_list"],
        request["h_base"],
        request["attr_values"],
        request["c1_prime"],
        request["c2_prime"],
        request["omegas"],
        request["c0_prime"]
    )

    # TA/AA partial decryption
    aggregate_partial = partial_decryption(
        request["ask_list"],
        request["h_base"],
        request["attr_values"],
        request["c1_hat"],
        request["c2_hat"],
        request["omegas"]
    )

    # Receiver final unblinding/key recovery
    final_unblinding_key_recovery(
        request["c0"],
        aggregate_partial,
        request["z"]
    )


# ============================================================
# Input generation
#
# Static synthetic inputs are generated outside timed regions.  Fresh one-time
# blinding factors z / z_j are sampled inside the baseline-online / offline-token
# operations, matching the scheme timing boundary.
# ============================================================

def make_baseline_online_request(l):
    return {
        "c1": [group.random(G1) for _ in range(l)],
        "c2": [group.random(G1) for _ in range(l)],
        "c1_prime": [group.random(G1) for _ in range(l)],
        "c2_prime": [group.random(G1) for _ in range(l)],
        "c0_prime": group.random(GT),
    }


def make_offline_request(l):
    return {
        "c1": [group.random(G1) for _ in range(l)],
        "c2": [group.random(G1) for _ in range(l)],
    }


def run_offline_request(request):
    # Includes fresh z_j, 2l E1, and token materialization.
    token = prepare_token(request["c1"], request["c2"])
    _ = token


def make_baseline_receiver_request(l):
    req = make_baseline_online_request(l)
    req["c0"] = group.random(GT)
    req["aggregate_partial"] = group.random(GT)
    return req


# ============================================================
# Prepared-token requests
#
# Every Ours request below receives an independently generated z_j and
# ciphertext-bound token. Token generation is completed before the timed
# online region and is measured separately by run_offline_request().
# ============================================================

def make_fresh_token_material(l):
    c1 = [group.random(G1) for _ in range(l)]
    c2 = [group.random(G1) for _ in range(l)]
    token = prepare_token(c1, c2)
    return {
        "z": token["z"],
        "c1_hat": tuple(token["c1_hat"]),
        "c2_hat": tuple(token["c2_hat"]),
    }


def make_ours_online_request(l):
    token = make_fresh_token_material(l)
    return {
        "z": token["z"],
        "c1_hat": token["c1_hat"],
        "c2_hat": token["c2_hat"],
        "c1_prime": tuple(group.random(G1) for _ in range(l)),
        "c2_prime": tuple(group.random(G1) for _ in range(l)),
        "c0_prime": group.random(GT),
        "used": False,
    }


def make_ours_receiver_request(l):
    req = make_ours_online_request(l)
    req["c0"] = group.random(GT)
    req["aggregate_partial"] = group.random(GT)
    return req


def make_baseline_path_request(l):
    return {
        "l": l,
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
    }


def make_ours_path_request(l):
    req = make_ours_online_request(l)
    req.update({
        "l": l,
        "ask_list": [group.random(G1) for _ in range(l)],
        "h_base": group.random(G1),
        "attr_values": [group.random(ZR) for _ in range(l)],
        "omegas": [group.random(ZR) for _ in range(l)],
        "c0": group.random(GT),
    })
    return req


# ============================================================
# Batch measurement
#
# Each returned value is per-request wall-clock latency.  Input
# generation is completed before timing.  GC is collected before
# each timed batch and disabled only while that batch is measured,
# matching the intent of the Android timing code and reducing
# interpreter-GC noise without moving any cryptographic operation
# across the measurement boundary.
# ============================================================

def measure_batch(requests, request_function):
    if not requests:
        raise ValueError("measure_batch() received an empty request batch")

    gc.collect()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()

    try:
        start = time.perf_counter_ns()
        for request in requests:
            request_function(request)
        end = time.perf_counter_ns()
    finally:
        if was_enabled:
            gc.enable()

    total_ms = (end - start) / 1_000_000.0
    return total_ms / len(requests)


def make_batch(make_request, batch_size):
    """Prepare all inputs outside the timed region."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [make_request() for _ in range(batch_size)]


def collect_counterbalanced_group(
    metric_specs,
    orders,
    batch_size,
    warmup,
    iterations,
    l_value,
    family_name,
    order_log,
):
    """
    Measure several metrics in a counterbalanced order.

    metric_specs maps metric name -> (make_request, request_function).
    Every metric uses the same batch_size.  Warm-up batches are executed
    but not retained.  Measured-order cycling restarts after warm-up so
    warm-up count cannot skew the measured order balance.
    """
    if not metric_specs:
        raise ValueError("metric_specs must not be empty")
    if not orders:
        raise ValueError("orders must not be empty")

    metric_names = set(metric_specs)
    for order in orders:
        if set(order) != metric_names or len(order) != len(metric_specs):
            raise ValueError(
                f"Invalid order {order!r} for metrics {sorted(metric_names)!r}"
            )

    samples = {name: [] for name in metric_specs}
    total = warmup + iterations

    for iteration in range(total):
        measured = iteration >= warmup
        measured_index = iteration - warmup if measured else iteration
        order = orders[measured_index % len(orders)]

        # Prepare ALL metric inputs before starting any timed batch in this
        # iteration, so request construction is outside every timed region.
        batches = {
            name: make_batch(metric_specs[name][0], batch_size)
            for name in metric_specs
        }

        result = {}
        for name in order:
            request_function = metric_specs[name][1]
            result[name] = measure_batch(batches[name], request_function)

        if measured:
            for name in metric_specs:
                samples[name].append(result[name])

            order_log.append({
                "l": l_value,
                "family": family_name,
                "measurement": measured_index + 1,
                "order": "->".join(order),
                "batch_size": batch_size,
            })

    for name, values in samples.items():
        if len(values) != iterations:
            raise RuntimeError(
                f"Expected {iterations} samples for {family_name}/{name}, "
                f"got {len(values)}"
            )

    return samples


def save_order_log(rows, path=ORDER_LOG_FILE):
    fields = ["l", "family", "measurement", "order", "batch_size"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(path=METADATA_FILE):
    try:
        script_path = Path(__file__).resolve()
        script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    except Exception:
        script_path = None
        script_sha256 = "unavailable"

    lines = [
        "CPU operation-level benchmark metadata",
        f"timestamp_local={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"script_path={script_path if script_path is not None else 'unavailable'}",
        f"script_sha256={script_sha256}",
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"pid={os.getpid()}",
        "pairing_group=SS512",
        f"l_values={ATTRIBUTES}",
        f"warmup_batches_per_metric={WARMUP}",
        f"measured_batches_per_metric={ITERATIONS}",
        f"reenc_batch_size={REENC_BATCH}",
        f"receiver_batch_size={RECEIVER_BATCH}",
        f"path_batch_size={PATH_BATCH}",
        "cpu_summary_trim_each_tail=0.10",
        "candidate_set_enumeration=excluded",
        "lsss_coefficient_solving=excluded",
        "input_generation=outside_timed_region",
        "measurement_order=counterbalanced_within_each_metric_family",
        "token_pool_lookup=excluded",
        "single_thread_token_state_transition=modeled_in_memory",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main experiment
# ============================================================

def validate_configuration():
    if WARMUP < 0 or ITERATIONS <= 0:
        raise ValueError("WARMUP must be >= 0 and ITERATIONS must be > 0")
    if any((not isinstance(l, int)) or l <= 0 for l in ATTRIBUTES):
        raise ValueError("ATTRIBUTES must contain positive integers")
    if len(set(ATTRIBUTES)) != len(ATTRIBUTES):
        raise ValueError("ATTRIBUTES contains duplicate policy sizes")
    if min(REENC_BATCH, RECEIVER_BATCH, PATH_BATCH) <= 0:
        raise ValueError("All batch sizes must be positive")


def simulate_and_plot():
    validate_configuration()

    results = {

        "offline_ours": {
            "mean": [],
            "std": []
        },

        "online_base": {
            "mean": [],
            "std": []
        },

        "online_ours": {
            "mean": [],
            "std": []
        },

        "receiver_base": {
            "mean": [],
            "std": []
        },

        "receiver_ours": {
            "mean": [],
            "std": []
        },

        "path_base": {
            "mean": [],
            "std": []
        },

        "path_ours": {
            "mean": [],
            "std": []
        }
    }

    raw_results = {}

    print(
        "Start Charm-Crypto batched latency experiment..."
    )

    print(
        "Run without cpulimit: CPU results are controlled operation-level scaling benchmarks."
    )
    print(
        "TA/AA timing follows the baseline operation sequence for one selected set |I|=l; "
        "candidate enumeration and LSSS solving are excluded. Ours uses independent single-use tokens."
    )
    print(
        "Within each metric family, all schemes use the same fixed batch size and "
        "measurement order is counterbalanced."
    )

    print(
        "l values:",
        ATTRIBUTES
    )

    # ========================================================
    # l loop
    # ========================================================

    order_log = []
    save_metadata()

    for l in ATTRIBUTES:

        print()
        print(f"Testing l = {l}")
        print(f"  reenc batch = {REENC_BATCH} (shared by baseline/offline/ours)")
        print(f"  receiver batch = {RECEIVER_BATCH} (shared by baseline/ours)")
        print(f"  path batch = {PATH_BATCH} (shared by baseline/ours)")

        # ----------------------------------------------------
        # A. One-time re-encryption family
        #
        # Baseline online, Ours offline and Ours online use
        # exactly the same fixed batch size and all six order
        # permutations are cycled.
        # ----------------------------------------------------

        reenc_samples = collect_counterbalanced_group(
            metric_specs={
                "online_base": (
                    lambda l=l: make_baseline_online_request(l),
                    baseline_online_request,
                ),
                "offline_ours": (
                    lambda l=l: make_offline_request(l),
                    run_offline_request,
                ),
                "online_ours": (
                    lambda l=l: make_ours_online_request(l),
                    ours_online_request,
                ),
            },
            orders=REENC_ORDERS,
            batch_size=REENC_BATCH,
            warmup=WARMUP,
            iterations=ITERATIONS,
            l_value=l,
            family_name="reenc",
            order_log=order_log,
        )

        # ----------------------------------------------------
        # B. Receiver-side online cryptographic path
        #
        # Baseline/Ours alternate B->O and O->B.
        # ----------------------------------------------------

        receiver_group = collect_counterbalanced_group(
            metric_specs={
                "baseline": (
                    lambda l=l: make_baseline_receiver_request(l),
                    baseline_receiver_request,
                ),
                "ours": (
                    lambda l=l: make_ours_receiver_request(l),
                    ours_receiver_request,
                ),
            },
            orders=PAIR_ORDERS,
            batch_size=RECEIVER_BATCH,
            warmup=WARMUP,
            iterations=ITERATIONS,
            l_value=l,
            family_name="receiver",
            order_log=order_log,
        )

        # ----------------------------------------------------
        # C. Online decryption-path cryptographic path
        #
        # Baseline/Ours alternate B->O and O->B.
        # ----------------------------------------------------

        path_group = collect_counterbalanced_group(
            metric_specs={
                "baseline": (
                    lambda l=l: make_baseline_path_request(l),
                    baseline_path_request,
                ),
                "ours": (
                    lambda l=l: make_ours_path_request(l),
                    ours_path_request,
                ),
            },
            orders=PAIR_ORDERS,
            batch_size=PATH_BATCH,
            warmup=WARMUP,
            iterations=ITERATIONS,
            l_value=l,
            family_name="path",
            order_log=order_log,
        )

        offline_samples = reenc_samples["offline_ours"]
        online_base_samples = reenc_samples["online_base"]
        online_ours_samples = reenc_samples["online_ours"]
        receiver_base_samples = receiver_group["baseline"]
        receiver_ours_samples = receiver_group["ours"]
        path_base_samples = path_group["baseline"]
        path_ours_samples = path_group["ours"]

        metric_samples = {

            "offline_ours":
                offline_samples,

            "online_base":
                online_base_samples,

            "online_ours":
                online_ours_samples,

            "receiver_base":
                receiver_base_samples,

            "receiver_ours":
                receiver_ours_samples,

            "path_base":
                path_base_samples,

            "path_ours":
                path_ours_samples
        }

        raw_results[l] = (
            metric_samples
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        for key, samples in (
            metric_samples.items()
        ):

            mean, std = (
                trimmed_mean_std(
                    samples,
                    trim_ratio=0.1
                )
            )

            results[key][
                "mean"
            ].append(mean)

            results[key][
                "std"
            ].append(std)

        # Checkpoint after the full setting is complete. This I/O is outside
        # all timed regions and protects completed measurements from interruption.
        save_partial_raw(raw_results)

        print(
            "  completed (checkpoint saved)"
        )

    expected_order_rows = len(ATTRIBUTES) * ITERATIONS * 3
    if len(order_log) != expected_order_rows:
        raise RuntimeError(
            f"Order-log length mismatch: expected {expected_order_rows}, "
            f"got {len(order_log)}"
        )

    for l, metric_samples in raw_results.items():
        for metric_name, values in metric_samples.items():
            if len(values) != ITERATIONS:
                raise RuntimeError(
                    f"Sample-count mismatch for l={l}, {metric_name}: "
                    f"expected {ITERATIONS}, got {len(values)}"
                )
            if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                raise RuntimeError(
                    f"Non-finite latency detected for l={l}, {metric_name}"
                )

    # ========================================================
    # Save raw batch-level measurements
    # ========================================================

    with open(
        "latency_raw_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "l",
            "measurement",
            "offline_ours_ms",
            "online_base_ms",
            "online_ours_ms",
            "receiver_base_ms",
            "receiver_ours_ms",
            "path_base_ms",
            "path_ours_ms"
        ])

        for l in ATTRIBUTES:

            samples = (
                raw_results[l]
            )

            for i in range(
                ITERATIONS
            ):

                writer.writerow([
                    l,
                    i + 1,

                    samples[
                        "offline_ours"
                    ][i],

                    samples[
                        "online_base"
                    ][i],

                    samples[
                        "online_ours"
                    ][i],

                    samples[
                        "receiver_base"
                    ][i],

                    samples[
                        "receiver_ours"
                    ][i],

                    samples[
                        "path_base"
                    ][i],

                    samples[
                        "path_ours"
                    ][i]
                ])

    # Save the measured execution order for audit/reproducibility.
    save_order_log(order_log)

    # ========================================================
    # Save summary
    # ========================================================

    with open(
        "latency_summary_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        header = ["l"]

        for key in results:

            header += [
                f"{key}_mean_ms",
                f"{key}_std_ms"
            ]

        writer.writerow(
            header
        )

        for idx, l in enumerate(
            ATTRIBUTES
        ):

            row = [l]

            for key in results:

                row += [

                    results[key][
                        "mean"
                    ][idx],

                    results[key][
                        "std"
                    ][idx]
                ]

            writer.writerow(
                row
            )

    # ========================================================
    # Diagnostic vector PDF (publication plots may instead be regenerated
    # from latency_summary_results.csv using the paper-specific drawing script).
    # ========================================================

    plt.rcParams[
        "pdf.fonttype"
    ] = 42

    plt.rcParams[
        "ps.fonttype"
    ] = 42

    plt.rcParams.update({

        "font.family":
            "serif",

        "font.size":
            11,

        "axes.labelsize":
            11,

        "axes.titlesize":
            12,

        "legend.fontsize":
            10
    })

    x = np.array(
        ATTRIBUTES
    )

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.65)
    )

    # ========================================================
    # (a) One-time re-encryption
    # ========================================================

    ax = axs[0]

    ax.errorbar(
        x,

        results[
            "online_base"
        ]["mean"],

        yerr=results[
            "online_base"
        ]["std"],

        fmt="--o",
        capsize=3,

        label=
            "Baseline online re-enc."
    )

    ax.errorbar(
        x,

        results[
            "offline_ours"
        ]["mean"],

        yerr=results[
            "offline_ours"
        ]["std"],

        fmt="-.s",
        capsize=3,

        label=
            "Ours offline token gen."
    )

    ax.errorbar(
        x,

        results[
            "online_ours"
        ]["mean"],

        yerr=results[
            "online_ours"
        ]["std"],

        fmt="-^",
        capsize=3,

        label=
            "Ours online re-enc."
    )

    ax.set_title(
        "(a) One-time re-encryption"
    )

    ax.set_xlabel(
        "Number of attributes / LSSS rows"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_xticks(x)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    # ========================================================
    # (b) Receiver-side online cryptographic latency
    # ========================================================

    ax = axs[1]

    ax.errorbar(
        x,

        results[
            "receiver_base"
        ]["mean"],

        yerr=results[
            "receiver_base"
        ]["std"],

        fmt="--o",
        capsize=3,

        label=
            "Baseline receiver-side"
    )

    ax.errorbar(
        x,

        results[
            "receiver_ours"
        ]["mean"],

        yerr=results[
            "receiver_ours"
        ]["std"],

        fmt="-s",
        capsize=3,

        label=
            "Ours receiver-side"
    )

    ax.set_title(
        "(b) Receiver-side online\ncryptographic latency"
    )

    ax.set_xlabel(
        "Number of attributes / LSSS rows"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_xticks(x)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    # ========================================================
    # (c) Online decryption-path cryptographic latency
    # ========================================================

    ax = axs[2]

    ax.errorbar(
        x,

        results[
            "path_base"
        ]["mean"],

        yerr=results[
            "path_base"
        ]["std"],

        fmt="--o",
        capsize=3,

        label=
            "Baseline online path"
    )

    ax.errorbar(
        x,

        results[
            "path_ours"
        ]["mean"],

        yerr=results[
            "path_ours"
        ]["std"],

        fmt="-s",
        capsize=3,

        label=
            "Ours online path"
    )

    ax.set_title(
        "(c) Online decryption-path\ncryptographic latency"
    )

    ax.set_xlabel(
        "Number of attributes / LSSS rows"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_xticks(x)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    plt.tight_layout()

    output_pdf = (
        "Fig2_CPU_Latency_from_final_run.pdf"
    )

    fig.savefig(
        output_pdf,
        format="pdf"
    )

    plt.close(fig)

    print()
    print(
        "Experiment completed."
    )

    print(
        "Files saved:"
    )

    print(
        " ",
        output_pdf
    )

    print(
        "  latency_raw_results.csv"
    )

    print(
        "  latency_summary_results.csv"
    )

    print(
        "  latency_raw_results_partial.csv"
    )

    print(
        " ",
        ORDER_LOG_FILE
    )

    print(
        " ",
        METADATA_FILE
    )


if __name__ == "__main__":
    simulate_and_plot()
