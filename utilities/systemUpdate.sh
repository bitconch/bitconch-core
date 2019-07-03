#!/usr/bin/env bash
set -ex




update() {
	echo -e "Start to update && upgrade"
	if [ $PKG = "Ubuntu" ] ; then
                sudo apt update -y && sudo apt upgrade -y
        elif [ $PKG = "Debian" ] ; then
                yum apt update -y && sudo apt upgrade -y
        fi
	echo -e "Update && upgrade finished"
}
update