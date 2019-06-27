#!/usr/bin/env python3
"""
Will deploy nightly Bitconch chain, codename soros 
"""
import logging
import stat
import shutil
import os, re, argparse, sys,crypt
import getpass
import click
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


def execute_shell(command, silent=False, cwd=None, shell=True, env=None):
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
    execute_shell("git submodule update --init --recursive", silent=False)

    # Fetch upstream changes
    execute_shell("git submodule foreach --recursive git fetch ", silent=False)

    # Reset to upstream
    execute_shell("git submodule foreach git reset --hard origin/HEAD", silent=False)

    # Update include/
    if os.path.exists("include"):
        prnt_run("Clean the include folder")
        rmtree("include",onerror=rmtree_onerror)
        prnt_run("Copy the latest header file from vendor/rustelo-rust/include")
    copytree("vendor/rustelo-rust/include", "include")


 


def build(rust_version,cargoFeatures,release=False):
    target_list = execute_shell("rustup target list", silent=True).decode()
    m = re.search(r"(.*?)\s*\(default\)", target_list)
    
    #currentWorking directory
    pwd = os.getcwd()
    
    default_target =m[1]

    # building priority:
    # 1. x86_64-pc-windows-gnu  for 64-bit MinGW (Windows 7+)
    # 2. x86_64-unknown-linux-musl for linux musl
    # 3. x86_64-unknown-linux-musl for linux ubuntu,debian
    # 4. x86_64-apple-darwin for macOS-10


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
                execute_shell(["rustup", "target", "add", target],
                   shell=False,
                   silent=True,
                   #cwd="vendor/rustelo-rust")
                   cwd="buffett2")

            profile = "--release" if release else ''
            execute_shell(f"cargo build --all --target {target} {profile}",
               #cwd="vendor/rustelo-rust",
               cwd="buffett2",
               env={
                   "CC": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER": f"{prefix[target]}gcc",
                   "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER": f"{prefix[target]}gcc",
               })

            if target.endswith("-apple-darwin"):
                execute_shell(f"strip -Sx {artifact[target]}",
                   cwd=f"vendor/rustelo-rust/soros/target/{target}/release", silent=True)

            else:
                execute_shell(f"{prefix[target]}strip --strip-unneeded -d -x {artifact[target]}",
                   #cwd=f"vendor/rustelo-rust/target/{target}/release")
                   cwd=f"vendor/rustelo-rust/soros/target/{target}/release")

            #copy2(f"vendor/rustelo-rust/target/{target}/release/{artifact[target]}", f"libs/{target}/")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-fullnode", f"libs/{target}/buffett-fullnode")
            #copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-fullnode-config", f"libs/{target}/buffett-fullnode-config")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-drone", f"libs/{target}/buffett-drone")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-bench-tps", f"libs/{target}/buffett-bench-tps")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-ledger-tool", f"libs/{target}/buffett-ledger-tool")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-genesis", f"libs/{target}/buffett-genesis")
            copy2(f"vendor/rustelo-rust/soros/target/{target}/release/soros-keygen", f"libs/{target}/buffett-keygen")

    else:
        target = default_target

        # For development; build only the _default_ target
        prnt_run(f"Build the rust+c code in soros for {target}")

        execute_shell(f"cargo build --all --release --features=erasure", cwd="vendor/rustelo-rust/soros")
        
        # Copy _default_ lib over
        
        if not os.path.exists(f"libs/{target}/"):
            prnt_run(f"Check the lib folder, if not , create one ")
            os.makedirs(f"libs/{target}/")
        if not os.path.exists(f"libs/{target}/bin/deps"):
            prnt_run(f"Check the the depslib folder, if not , create one ")
            os.makedirs(f"libs/{target}/bin/deps/")

        prnt_run(f"Copy the generated artifact file and dependencies")

        BIN_CRATES=[
            "bench-tps",
            "drone",
            "fullnode",
            "genesis",
            "gossip",
            "install",
            "keygen",
            "ledger-tool",
            "wallet"
        ]

        for crate in BIN_CRATES:
            # execute_shell(f"set -x && cargo  install --force --path '{crate}' --root '{pwd}/libs/{target}/'  --features='erasure'", cwd="vendor/rustelo-rust/soros")
            execute_shell(f"set -x && cargo  install --force --path '{crate}' --root '{pwd}/libs/{target}/'  --features='erasure'", cwd="vendor/rustelo-rust/soros")

        # copy dependencies into deps folder
        # execute_shell(f"set -x && cp *.so {pwd}/libs/{target}/bin/deps",cwd="vendor/rustelo-rust/soros/target/release")
        execute_shell(f"set -x && cp libsoros*.so {pwd}/libs/{target}/bin/deps",cwd="vendor/rustelo-rust/soros/target/release/deps")


    deploy_bin(target)



def commit():
    sha = execute_shell("git rev-parse --short HEAD", cwd="vendor/rustelo-rust", silent=True).decode().strip()
    execute_shell("git add ./vendor/rustelo-rust ./libs ./include")

    try:
        execute_shell(f"git commit -m \"Build libs/ and sync include/ from rustelo#{sha}\"")
        execute_shell("git push")

    except CalledProcessError:
        # Commit likely failed because there was nothing to commit
        pass



def createUser(name,username, password):
    prnt_run(f"Create a new user")
    encPass =crypt.crypt(password,"22")
    return os.system("useradd -p"+encPass+"-s"+"/bin/bash"+"-d"+"/home/"+username+"-m"+"-c \""+name +"\""+username)


def deploy_bin(target):
    # installation location /usr/bin/bitconch
    # remove previous installed version
    if os.path.exists("/usr/bin/bitconch"):
        prnt_run("Remove previous installed version")
        rmtree("/usr/bin/bitconch",onerror=rmtree_onerror)
        prnt_run("Copy the compiled binaries to /usr/bin/bitconch")
    # cp the binary into the folder 
    copytree(f"libs/{target}/", "/usr/bin/bitconch")
    
    # seth PATH variable 
    prnt_run(f"Set PATH to include soros executables ")
    execute_shell("echo 'export PATH=/usr/bin/bitconch/bin:$PATH' >>~/.profile")
    execute_shell("echo 'export PATH=/usr/bin/bitconch/bin/deps:$PATH' >>~/.profile")
    # execute_shell("source ~/.profile")

    # remove the previous installed service file
    if os.path.exists("/etc/systemd/system/soros-leader.service"):
        prnt_run("Remove previous installed service file:soros-leader.service")
        os.remove("/etc/systemd/system/soros-leader.service")
    if os.path.exists("/etc/systemd/system/soros-leader.socket"):
        prnt_run("Remove previous installed socket file:soros-leader.socket")
        os.remove("/etc/systemd/system/soros-leader.socket")

    if os.path.exists("/etc/systemd/system/soros-tokenbot.service"):
        prnt_run("Remove previous installed service file:soros-tokenbot.service")
        os.remove("/etc/systemd/system/soros-tokenbot.service")
    if os.path.exists("/etc/systemd/system/soros-tokenbot.socket"):
        prnt_run("Remove previous installed socket file:soros-tokenbot.socket")
        os.remove("/etc/systemd/system/soros-tokenbot.socket")

    if os.path.exists("/etc/systemd/system/soros-validator.service"):
        prnt_run("Remove previous installed service file:soros-validator.service")
        os.remove("/etc/systemd/system/soros-validator.service")
    if os.path.exists("/etc/systemd/system/soros-validator.socket"):
        prnt_run("Remove previous installed socket file:soros-validator.socket")
        os.remove("/etc/systemd/system/soros-validator.socket")

    # cp the service files into service folder
    execute_shell("cp soros.service.template/*  /etc/systemd/system")

    # create the working directory data directory
    copytree(f"soros.scripts/demo", "/usr/bin/bitconch/soros/demo")
    copytree(f"soros.scripts/scripts", "/usr/bin/bitconch/soros/scripts")

   
parser = argparse.ArgumentParser()
parser.add_argument(
    "-R", "--release", help="build in release mode", action="store_true")
parser.add_argument(
    "-C", "--commit", help="commit include/ and libs/", action="store_true")

argv = parser.parse_args(sys.argv[1:])

update_submodules()
build("1.35","erasure",release=argv.release)
prnt_run("Update PATH")
# execute_shell(f"source ~/.profile")
prnt_run("Please run /usr/bin/bitconch/soros/demo/setup.sh")
if click.confirm('Do you want to run setup to create genesis file and id files?', default=True):
    execute_shell("/usr/bin/bitconch/soros/demo/setup.sh",cwd="/usr/bin/bitconch/soros")
#createUser("billy","billy","123456")
# create a bin folder at /usr/bin/bitconch
# prnt_run(getpass.getuser())
if click.confirm('Do you want to reload the systemctl daemon?', default=True):
    execute_shell("systemctl daemon-reload")

if click.confirm('Are you running on the leader node?', default=True):
    # backup the existing rsync configuration file
    if os.path.exists("/etc/rsyncd.conf"):
        prnt_run("Backup the existing rsyncd.conf.")
        copy2(f"/etc/rsyncd.conf", f"/etc/rsyncd.conf.bk")
        os.remove("/etc/rsyncd.conf")
    prnt_run("Setup new rsyncd.conf.")
    copy2(f"rsyncd.conf", f"/etc/rsyncd.conf")
    execute_shell("systemctl enable rsync")
    execute_shell("systemctl start rsync")

if argv.commit and argv.release:
    commit()

