#!/usr/bin/env bash
set -ex

check_os() {
	if [[ "$(uname -v)" == *"Linux"* ]] ; then
		PKG="linux"   # linux is my default 
		echo "Running on Linux"
	elif [[ "$(uname -v)" == *"Ubuntu"* ]] ; then
		PKG="Ubuntu"
		echo "Running on Ubuntu"
	else
		echo "Unknown operating system"
		echo "Please select your operating system"
		echo "Choices:"
		echo "	     linux - any linux distro"
		echo "	     darwin - MacOS"
		read PKG
	fi
}

check_os