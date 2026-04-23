"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) entry point."""

from optimizers.cmaes.sampler import build
from optimizers.common.cli import build_parser
from optimizers.common.driver import run


def main() -> None:
    args = build_parser("cmaes").parse_args()
    run("cmaes", build(), args)


if __name__ == "__main__":
    main()
