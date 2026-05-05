from __future__ import annotations

from risk_controlled_otta.experiments.single_domain_step_history_common import (
    parse_common_args,
    run_single_domain_history,
)


def main() -> None:
    args = parse_common_args()
    run_single_domain_history(args, method_name="ours")


if __name__ == "__main__":
    main()


