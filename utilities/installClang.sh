
#!/usr/bin/env bash
set -ex

install_clang() {
	echo "Start to install clang"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install clang -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install clang -y
	fi
	clang -v
	echo "clang installed"
}
install_clang