import argparse
import sys


import deepview_profile
import deepview_profile.commands.interactive
import deepview_profile.commands.memory
import deepview_profile.commands.time


def main():
    parser = argparse.ArgumentParser(
        prog = deepview_profile.__name__,
        description = deepview_profile.__description__
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Print the version and exit.",
    )
    subparsers = parser.add_subparsers(title="Commands")
    deepview_profile.commands.interactive.register_command(subparsers)
    deepview_profile.commands.memory.register_command(subparsers)
    deepview_profile.commands.time.register_command(subparsers)
    args = parser.parse_args()

    if args.version:
        print('DeepView Command Line Interface', 'v' + deepview_profile.__version__,)
        return

    if 'func' not in args:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == '__main__':
    main()
