"""Random sampler entry point — a sanity baseline."""

from optimizers.common.cli import build_parser
from optimizers.common.driver import run
from optimizers.random.sampler import build


def main() -> None:
    args = build_parser("random").parse_args()
    run("random", build(), args)


if __name__ == "__main__":
    main()
