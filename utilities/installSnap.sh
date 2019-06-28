#!/usr/bin/env bash
set -ex

install_SNAP() {
        echo "Start to install SNAP"
        if [ $PKG = "Ubuntu" ] ; then
		sudo apt install snapd
        elif [ $PKG = "Debian" ] ; then
                yum apt install snapd
        fi
        snap version
        echo "SNAP installed"
}


install_SNAP