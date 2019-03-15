#!/usr/bin/env python3
"""
Use Python3(>=3.6.7) and Shell to build  

1. 
"""
import logging
import stat
import shutil
import os, re, argparse, sys
from subprocess import Popen, check_call, PIPE, check_output, CalledProcessError
from shutil import copy2, copytree, rmtree
from colorama import init
init()
from colorama import Fore, Back, Style


def rmtree_onerror(self, func, file_path, exc_info):
    """
    Error handler for ``shutil.rmtree``.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    If the error is for another reason it re-raises the error.
    Usage : ``shutil.rmtree(path, onerror=onerror)`` 
    """
    logging.warning(str(exc_info))
    logging.warning("rmtree error,check the file exists or try to chmod the file,then retry rmtree action.")
    os.chmod(file_path, stat.S_IWRITE) #chmod to writeable
    if os.path.isdir(file_path):
        #file exists
       func(file_path)
    else:
        #handle whatever
        raise


def sub01_execute_shell(command, silent=False, cwd=None, shell=True, env=None):
    """
    Execute a system command 
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

def prnt_warn(in_text):
    """
    Print a warning message
    """
    print(Fore.YELLOW + "[!]"+in_text)
    print(Style.RESET_ALL)

def prnt_run(in_text):
    """
    Print a processing message
    """
    print(Fore.WHITE + "[~]"+in_text)
    print(Style.RESET_ALL)

def prnt_error(in_text):
    """
    Print an error message
    """
    print(Fore.RED + "[~]"+in_text)
    print(Style.RESET_ALL)


def update_submodules():
    """
    Pull the latest submodule code from upstream
    """
    prnt_warn('This repo uses submodules to manage the codes')
    prnt_run("Use git to update the submodules")
    # Ensure the submodule is initialized
    sub01_execute_shell("git submodule update --init --recursive", silent=False)

    # Fetch upstream changes
    sub01_execute_shell("git submodule foreach --recursive git fetch ", silent=False)

    # Reset to upstream
    sub01_execute_shell("git submodule foreach git reset --hard origin/HEAD", silent=False)

    # Update include/
    if os.path.exists("include"):
        prnt_run("Clean the include folder")
        rmtree("include",onerror=rmtree_onerror)
        prnt_run("Copy the latest header file from vendor/rustelo-rust/include")
    copytree("vendor/rustelo-rust/include", "include")


 


def build(release=False):
    target_list = sub01_execute_shell("rustup target list", silent=True).decode()
    m = re.search(r"(.*?)\s*\(default\)", target_list)

    default_target =m[1]

    # building priority:
    # 1. x86_64-pc-windows-gnu  for 64-bit MinGW (Windows 7+)
    # 2. x86_64-unknown-linux-musl for 64-bit MSVC (Windows 7+)
    # 3. 


    target_list = [
        "x86_64-pc-windows-gnu",
        "x86_64-unknown-linux-musl",
        "x86_64-unknown-linux-gnu",
        "x86_64-apple-darwin"
    ]

    prefix = {
        "x86_64-pc-windows-gnu": "x86_64-w64-mingw32-",
        "x86_64-unknown-linux-musl": "x86_64-linux-musl-",
        "x86_64-unknown-linux-gnu": "x86_64-linux-gnu-",
         "x86_64-apple-darwin": ""
    }

    artifact = {
        "x86_64-pc-windows-gnu": "rustelo.dll",
        "x86_64-unknown-linux-musl": "librustelo.so",
        "x86_64-unknown-linux-gnu": "librustelo.so",
        "x86_64-apple-darwin": "librustelo.dylib"
    }

    if release:
        for target in target_list:
            prnt_run(f"Build rust source for {target}")

            if target != default_target:
                sub01_execute_shell(["rustup", "target", "add", target],
                   shell=False,
                   silent=True,
                   cwd="vendor/rustelo-rust")

            profile = "--release" if release else ''
            sub01_execute_shell(f"cargo build --target {target} {profile}",
               cwd="vendor/rustelo-rust",
               env={
                   "CC": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER": f"{prefix[target]}gcc",
               })

            if target.endswith("-apple-darwin"):
                sub01_execute_shell(f"strip -Sx {artifact[target]}",
                   cwd=f"vendor/rustelo-rust/target/{target}/release", silent=True)

            else:
                sub01_execute_shell(f"{prefix[target]}strip --strip-unneeded -d -x {artifact[target]}",
                   cwd=f"vendor/rustelo-rust/target/{target}/release")

            copy2(f"vendor/rustelo-rust/target/{target}/release/{artifact[target]}", f"libs/{target}/")

    else:
        target = default_target

        # For development; build only the _default_ target
        prnt_run(f"build the rust+c code in vendor/rustelo-rust for {target}")
        sub01_execute_shell(f"cargo build  --target {target}", cwd="vendor/rustelo-rust/buffett")
        sub01_execute_shell(f"cargo build  --target {target}", cwd="vendor/rustelo-rust")

        # Copy _default_ lib over
        prnt_run(f"check the lib folder, if not , create one ")
        if not os.path.exists(f"libs/{target}/"):
            os.makedirs(f"libs/{target}/")
        prnt_run(f"copy the generated artifact file")
        copy2(f"vendor/rustelo-rust/target/{target}/debug/{artifact[target]}", f"libs/{target}/")


def commit():
    sha = sub01_execute_shell("git rev-parse --short HEAD", cwd="vendor/rustelo-rust", silent=True).decode().strip()
    sub01_execute_shell("git add ./vendor/rustelo-rust ./libs ./include")

    try:
        sub01_execute_shell(f"git commit -m \"build libs/ and sync include/ from rustelo#{sha}\"")
        sub01_execute_shell("git push")

    except CalledProcessError:
        # Commit likely failed because there was nothing to commit
        pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "-R", "--release", help="build in release mode", action="store_true")
parser.add_argument(
    "-C", "--commit", help="commit include/ and libs/", action="store_true")

argv = parser.parse_args(sys.argv[1:])

update_submodules()
build(release=argv.release)

if argv.commit and argv.release:
    commit()
