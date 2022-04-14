import argparse
import enum
import sys

import toml

import skyline
import skyline.commands.interactive
import skyline.commands.memory
import skyline.commands.time


def main():
    package = toml.load("pyproject.toml")
    parser = argparse.ArgumentParser(
        prog = package["tool"]["poetry"]["name"],
        description = package["tool"]["poetry"]["description"]
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
        print('Skyline Command Line Interface', 'v' + package["tool"]["poetry"]["version"],)
        return

    if 'func' not in args:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == '__main__':
    main()
