import argparse
import enum
import sys

import skyline
import skyline.commands.interactive
import skyline.commands.memory
import skyline.commands.time


def main():
    parser = argparse.ArgumentParser(
        prog="skyline",
        description="Skyline: Interactive Neural Network Performance "
                    "Profiler, Visualizer, and Debugger for PyTorch",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Print the version and exit.",
    )
    subparsers = parser.add_subparsers(title="Commands")
    skyline.commands.interactive.register_command(subparsers)
    skyline.commands.memory.register_command(subparsers)
    skyline.commands.time.register_command(subparsers)
    args = parser.parse_args()

    if args.version:
        print('Skyline Command Line Interface', 'v' + skyline.__version__)
        return

    if 'func' not in args:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == '__main__':
    main()
