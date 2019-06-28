
#!/usr/bin/env bash
set -ex
install_pkgconfig() {
	echo "Start to install the pkg-config"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install pkg-config -y
		sudo apt install pkg-config -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install pkg-config -y
		yum apt install pkg-config -y
	fi
	apt-cache search pkg-config
	echo "pkg-config installed"
}

install_pkgconfig