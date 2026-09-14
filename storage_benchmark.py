import csv
import platform
import os
import statistics

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1

# ============================================================
# Android / Termux Storage Experiment
# Serialized ciphertext-bound token size
# ============================================================

group = PairingGroup('SS512')

# Policy sizes
l_values = [10, 20, 30, 40, 50]

# Number of precomputed tokens in the token pool
q = 5

# Number of repetitions for each l
repetitions = 50

# Output CSV file
csv_file = "storage_results.csv"

# ============================================================
# Device / experiment information
# ============================================================

print("# Device/Platform:", platform.platform())
print("# Python:", platform.python_version())
print("# Pairing group: SS512")
print("# PID:", os.getpid())
print("# token_pool_size_q:", q)
print("# repetitions:", repetitions)

print(
    "l,"
    "Mean_single_token_bytes,"
    "Std_single_token_bytes,"
    "Mean_single_token_KiB,"
    "Pool_q5_bytes,"
    "Pool_q5_KiB,"
    "ZR_bytes,"
    "G1_element_bytes"
)


# ============================================================
# Serialized token size
# ============================================================

def serialized_token_size(z, token_C1, token_C2):
    """
    Measure the serialized cryptographic payload of one
    ciphertext-bound token:

        T_j(CT) =
        (
            z_j,
            {C1_i^{z_j}, C2_i^{z_j}}_{i=1}^{l},
            state_j
        )

    Included:
      - serialized z_j
      - serialized 2l group elements
      - one-byte state flag

    Python object/list/dictionary overhead is excluded.
    """

    size = 0

    # z_j in ZR
    size += len(group.serialize(z))

    # l elements: C1_i^{z_j}
    for elem in token_C1:
        size += len(group.serialize(elem))

    # l elements: C2_i^{z_j}
    for elem in token_C2:
        size += len(group.serialize(elem))

    # state_j:
    # 0 = unused
    # 1 = used
    size += 1

    return size


# ============================================================
# Run experiment and save CSV
# ============================================================

with open(csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "l",
        "Mean_single_token_bytes",
        "Std_single_token_bytes",
        "Mean_single_token_KiB",
        "Pool_q5_bytes",
        "Pool_q5_KiB",
        "ZR_bytes",
        "G1_element_bytes"
    ])

    for l in l_values:

        # ----------------------------------------------------
        # Simulate ciphertext components:
        # CT contains {C1_i, C2_i}_{i=1}^{l}
        # ----------------------------------------------------

        C1 = [group.random(G1) for _ in range(l)]
        C2 = [group.random(G1) for _ in range(l)]

        token_sizes = []

        # ----------------------------------------------------
        # Generate and measure ciphertext-bound tokens
        # ----------------------------------------------------

        for _ in range(repetitions):

            z = group.random(ZR)

            token_C1 = [c ** z for c in C1]
            token_C2 = [c ** z for c in C2]

            size_bytes = serialized_token_size(
                z,
                token_C1,
                token_C2
            )

            token_sizes.append(size_bytes)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mean_token_bytes = statistics.mean(token_sizes)

        if len(token_sizes) > 1:
            std_token_bytes = statistics.stdev(token_sizes)
        else:
            std_token_bytes = 0.0

        # 1 KiB = 1024 bytes
        mean_token_kib = mean_token_bytes / 1024.0

        # ----------------------------------------------------
        # Token pool q = 5
        # ----------------------------------------------------

        pool_bytes = mean_token_bytes * q
        pool_kib = pool_bytes / 1024.0

        # ----------------------------------------------------
        # Primitive serialized element sizes
        # ----------------------------------------------------

        sample_z = group.random(ZR)
        sample_g = group.random(G1)

        zr_bytes = len(group.serialize(sample_z))
        g1_bytes = len(group.serialize(sample_g))

        # ----------------------------------------------------
        # Output row
        # ----------------------------------------------------

        row = [
            l,
            f"{mean_token_bytes:.2f}",
            f"{std_token_bytes:.2f}",
            f"{mean_token_kib:.4f}",
            f"{pool_bytes:.2f}",
            f"{pool_kib:.4f}",
            zr_bytes,
            g1_bytes
        ]

        writer.writerow(row)

        print(",".join(map(str, row)))


print("\n# Experiment completed.")
print("# Results saved to:", csv_file)