#!/usr/bin/env python3
import subprocess
import os, re, argparse, sys
from subprocess import Popen, check_call, PIPE, check_output, CalledProcessError
from shutil import copy2, copytree, rmtree

def get_default_target():
    targets = sh("rustup target list", silent=True).decode()
    m = re.search(r"(.*?)\s*\(default\)", targets)

    return m[1]

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

default_target = get_default_target()
targets = [
        "x86_64-apple-darwin", 
        "x86_64-pc-windows-gnu",
        "x86_64-unknown-linux-musl",
        "x86_64-pc-windows-msvc"
    ]

prefix = {
        "x86_64-apple-darwin": "",
        "x86_64-pc-windows-gnu": "x86_64-w64-mingw32-",
        "x86_64-unknown-linux-musl": "x86_64-linux-musl-",
        "x86_64-pc-windows-msvc": "x86_64-w64-msvc-"
    }

#refer to https://doc.rust-lang.org/reference/linkage.html
artifact = {
        "x86_64-apple-darwin": "libhellolib.dylib",
        "x86_64-pc-windows-gnu": "hellolib.dll",
        "x86_64-unknown-linux-musl": "libhellolib.so",
        "x86_64-pc-windows-msvc": "hellolib.dll"
    }
target = default_target
print(f">>>Clean the lib folder")
sh(f"del *", cwd="lib")
print(f">>>building rust code for {target}")
#sh(f"cargo build --target {target}", cwd="xi4win/vendor/rustcode")
sh(f"cargo build --release --target {target} ", cwd="rustcode/hello")
print(f"<<<building finished for {target}")
# Copy _default_ lib over
#copy2(f"xi4win/vendor/rustcode/target/{target}/debug/{artifact[target]}", f"xi4win/lib/{target}/")
print(f"<<<copying the artifact")
copy2(f"rustcode/hello/target/{target}/release/{artifact[target]}", f"lib/{target}/")

#sh(f"go build ", cwd="xi4win/gobin/cli")
#copy the executable and lib into the same folder
print(f">>>go build the source")
#sh(f"go build -o lib\{target}\main.exe -x -v -a cli")
subprocess.run(f"go build -o lib\{target}\main.exe -x -v -a github.com/bitconch/bus/cdylib/cli", shell=True)