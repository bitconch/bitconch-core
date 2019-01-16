import os, re, argparse, sys
from subprocess import Popen, check_call, PIPE, check_output, CalledProcessError
from shutil import copy2, copytree, rmtree


def sub01_execute_shell(command, silent=False, cwd=None, shell=True, env=None):
    """Execute an system command, using check_call 
    """
    if env is not None:
        env = dict(**os.environ, **env)

    if silent:
        p = Popen(
            command, shell=shell, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
        stdout, _ = p.communicate()

        return stdout

    else:
        check_call(command, shell=shell, cwd=cwd, env=env)






sha = sub01_execute_shell("git rev-parse --short HEAD", cwd="vendor/", silent=True).decode().strip()

print(sha)
