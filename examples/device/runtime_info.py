"""Apply pre-JAX runtime policy, then report the active backend with tinax.device."""

from tinax.device import configure_jax, device_info


def main() -> None:
    """Set runtime options before JAX starts, then summarize the visible devices."""
    configure_jax(preallocate=False, matmul_precision="high")

    info = device_info()
    print(f"backend={info.backend}")
    print(f"device_count={info.device_count} local_device_count={info.local_device_count}")
    print(f"process={info.process_index}/{info.process_count}")
    print(f"device_kinds={info.device_kinds}")


if __name__ == "__main__":
    main()
