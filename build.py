#!/usr/bin/env python3
"""
Use Python3.5 and Shell to build  

"""
import os, re, argparse, sys
from subprocess import Popen, check_call, PIPE, check_output, CalledProcessError
from shutil import copy2, copytree, rmtree



def sh(command, silent=False, cwd=None, shell=True, env=None):
    if env is not None:
        env = dict(**os.environ, **env)

    if silent:
        p = Popen(
            command, shell=shell, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
        stdout, _ = p.communicate()

        return stdout

    else:
        check_call(command, shell=shell, cwd=cwd, env=env)


def update_submodules():
    print(" :: use git to fetch the latest source cdoes")

    # Ensure the submodule is initialized
    sh("git submodule update --init", silent=True)

    # Fetch upstream changes
    sh("git submodule foreach git fetch", silent=True)

    # Reset to upstream
    sh("git submodule foreach git reset --hard origin/HEAD", silent=True)

    # Update include/
    #rmtree("include")
    # copytree("vendor/rust_src/include", "include")


 


def build(release=False):
    target_list = sh("rustup target list", silent=True).decode()
    m = re.search(r"(.*?)\s*\(default\)", target_list)

    default_target =m[1]

    # building priority:
    # 1. x86_64-pc-windows-gnu  for 64-bit MinGW (Windows 7+)
    # 2. x86_64-unknown-linux-musl for 64-bit MSVC (Windows 7+)
    # 3. 


    target_list = [
        "x86_64-pc-windows-gnu",
        "x86_64-unknown-linux-musl",
        "x86_64-apple-darwin"
    ]

    prefix = {
        "x86_64-pc-windows-gnu": "x86_64-w64-mingw32-",
        "x86_64-unknown-linux-musl": "x86_64-linux-musl-",
         "x86_64-apple-darwin": ""
    }

    artifact = {
        "x86_64-pc-windows-gnu": "rusteleo.a",
        "x86_64-unknown-linux-musl": "libhedera.a",
        "x86_64-apple-darwin": "libhedera.a"
    }

    if release:
        for target in target_list:
            print(f" :: build rust source file for {target}")

            if target != default_target:
                sh(["rustup", "target", "add", target],
                   shell=False,
                   silent=True,
                   cwd="vendor/rust_src")

            profile = "--release" if release else ''
            sh(f"cargo build --target {target} {profile}",
               cwd="vendor/rust_src",
               env={
                   "CC": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER": f"{prefix[target]}gcc",
               })

            if target.endswith("-apple-darwin"):
                sh(f"strip -Sx {artifact[target]}",
                   cwd=f"vendor/rust_src/target/{target}/release", silent=True)

            else:
                sh(f"{prefix[target]}strip --strip-unneeded -d -x {artifact[target]}",
                   cwd=f"vendor/rust_src/target/{target}/release")

            copy2(f"vendor/rust_src/target/{target}/release/{artifact[target]}", f"libs/{target}/")

    else:
        target = default_target

        # For development; build only the _default_ target
        print(f" :: build the rust+c source code for {target}")
        sh(f"cargo build  --target {target}", cwd="vendor/rust_src")

        # Copy _default_ lib over
        copy2(f"vendor/rust_src/target/{target}/debug/{artifact[target]}", f"libs/{target}/")


def commit():
    sha = sh("git rev-parse --short HEAD", cwd="vendor/rust_src", silent=True).decode().strip()
    sh("git add ./vendor/rust_src ./libs ./include")

    try:
        sh(f"git commit -m \"build libs/ and sync include/ from hedera-sdk-c#{sha}\"")
        sh("git push")

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
