import sys


def print_analysis_error(error, file=sys.stderr):
    print(
        "Skyline encountered an error when profiling your model:",
        file=file,
    )
    print("->", str(error), file=file)

    if error.file_context is not None:
        if error.file_context.line_number is not None:
            message = (
                "This error occurred on line {} when processing {}.".format(
                    error.file_context.line_number,
                    error.file_context.file_path,
                )
            )
        else:
            message = "This error occurred when processing {}.".format(
                error.file_context.file_path,
            )
        print("->", message, file=file)
