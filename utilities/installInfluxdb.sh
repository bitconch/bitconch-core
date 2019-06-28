#!/usr/bin/env bash
set -ex
install_influxdb() {
    echo "Start to install influxdb"
        if [ $PKG = "Ubuntu" ] ; then
                curl -sL https://repos.influxdata.com/influxdb.key | sudo apt-key add -
		source /etc/lsb-release
		echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
		sudo apt-get install influxdb
		influxd config
        elif [ $PKG = "Debian" ] ; then
                curl -sL https://repos.influxdata.com/influxdb.key | sudo apt-key add -
		source /etc/os-release
		test $VERSION_ID = "7" && echo "deb https://repos.influxdata.com/debian wheezy stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
		test $VERSION_ID = "8" && echo "deb https://repos.influxdata.com/debian jessie stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
		sudo apt-get install influxdb
		influxd config
        fi
        influxd version
        echo "influxdb installed"
}

install_influxdb