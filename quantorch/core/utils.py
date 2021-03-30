import torch

def memory_as_readable_str(num_bits: int) -> str:
    r"""Generate a human-readable string for the memory size.
    1 KiB = 1024 B; we use the binary prefix (KiB) [1,2] instead of the decimal prefix
    (KB) to avoid any confusion with multiplying by 1000 instead of 1024.
    [1] https://en.wikipedia.org/wiki/Binary_prefix
    [2] https://physics.nist.gov/cuu/Units/binary.html
    """

    suffixes = ["B", "KiB", "MiB", "GiB"]
    num_bytes = num_bits / 8

    for i, suffix in enumerate(suffixes):
        rounded = num_bytes / (1024 ** i)
        if rounded < 1024:
            break

    return f"{rounded:,.2f} {suffix}"
