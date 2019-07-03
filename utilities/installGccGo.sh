
#!/usr/bin/env bash
set -ex
install_gccgo() {
	echo "Start to install gccgo"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install gccgo -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install gccgo -y
	fi
	gccgo -v
	echo "gccgo installed"
}

install_gccgo