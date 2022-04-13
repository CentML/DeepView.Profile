import contextlib
import os
import sys

from skyline.exceptions import exceptions_as_analysis_errors


@contextlib.contextmanager
def user_code_environment(script_root_path, project_root):
    """
    A combined context manager that activates all relevant context managers
    used when running user code.
    """
    with sys_path_root(script_root_path):
        with prevent_module_caching():
            with exceptions_as_analysis_errors(project_root):
                yield


@contextlib.contextmanager
def sys_path_root(script_root_path):
    """
    A context manager that sets sys.path[0] to the specified path on entry and
    then restores it after exiting the context manager.
    """
    # As per the Python documentation, sys.path[0] always stores the path to
    # the directory containing the Python script that was used to start the
    # Python interpreter. The contents of sys.path are used to resolve module
    # imports.
    #
    # When we run user code (e.g., the user's entry point file), we want to run
    # it as if it was being directly executed by the user from the shell. For
    # example:
    #
    #   $ python3 entry_point.py
    #
    # For this to work, we need to ensure that sys.path[0] is the path to the
    # directory containing the entry_point.py file. However if we use exec(),
    # sys.path[0] is set to the path of Skyline's command line executable.
    #
    # To fix this problem, we set sys.path[0] to the correct root path before
    # running the user's code and restore it to Skyline's script path after the
    # execution completes. Doing this is **very important** as it ensures that
    # imports work as expected inside the user's code. This context manager
    # should be used each time we execute user code because imports can exist
    # inside user-defined functions.
    #
    # Setting and then restoring sys.path[0] is better than just appending the
    # user's path to sys.path because we want to avoid accidentally importing
    # anything from the user's codebase.
    skyline_script_root = sys.path[0]
    try:
        sys.path[0] = script_root_path
        yield
    finally:
        sys.path[0] = skyline_script_root


@contextlib.contextmanager
def prevent_module_caching():
    """
    A context manager that prevents any imported modules from being cached
    after exiting.
    """
    try:
        original_modules = sys.modules.copy()
        yield
    finally:
        newly_added = {
            module_name for module_name in sys.modules.keys()
            if module_name not in original_modules
        }
        for module_name in newly_added:
            del sys.modules[module_name]
