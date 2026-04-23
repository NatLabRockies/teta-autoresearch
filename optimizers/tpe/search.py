"""TPE (Tree-structured Parzen Estimator) sampler entry point."""

from optimizers.common.cli import build_parser
from optimizers.common.driver import run
from optimizers.tpe.sampler import build


def main() -> None:
    args = build_parser("tpe").parse_args()
    run("tpe", build(), args)


if __name__ == "__main__":
    main()
