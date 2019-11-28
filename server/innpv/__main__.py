import argparse
import enum
import sys

import innpv.interactive


def main():
    parser = argparse.ArgumentParser(
        prog="innpv",
        description="INNPV: Interactive Neural Network Performance Visualizer "
                    "for PyTorch",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Print the version and exit.",
    )
    subparsers = parser.add_subparsers(title="Commands")
    innpv.interactive.register_command(subparsers)
    args = parser.parse_args()

    if args.version:
        print("v0.0.0")
        return

    if 'func' not in args:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == '__main__':
    main()
