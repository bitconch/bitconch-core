
#!/usr/bin/env bash
set -ex

install_python3() {
	echo "Start to install python3"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install python3
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install python3
	fi
	python3 -V
	echo "Python3 installed"
}

install_python3