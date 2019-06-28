#!/usr/bin/env bash
set -ex


install_golang() {
	echo "Start to install gccgo"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install golang
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install golang
	fi
	go version
	echo "gccgo installed"
}

install_golang