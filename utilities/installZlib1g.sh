
#!/usr/bin/env bash
set -ex

install_zlib1g_dev() {
	echo "Start to install the zlib1g-dev"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install zlib1g-dev -y
		sudo apt install zlib1g-dev -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install zlib1g-dev -y
		yum apt install zlib1g-dev -y
	fi
	apt-cache search zlib1g-dev
	echo "zlib1g-dev installed"
}

install_zlib1g_dev