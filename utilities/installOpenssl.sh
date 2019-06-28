
#!/usr/bin/env bash
set -ex

install_openssl() {
	echo "Start to install the dev package of openssl"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install libssl-dev -y
		sudo apt install libssl-dev -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install libssl-dev -y
		yum apt install libssl-dev -y
	fi
	apt-cache search libssl-dev
	echo "openssl installed"
}

install_openssl