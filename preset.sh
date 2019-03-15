#!/usr/bin/env bash
set -ex

check_os() {
	if [[ "$(uname -v)" == *"Linux"* ]] ; then
		PKG="linux"   # linux is my default 
		echo "Running on Linux"
	elif [[ "$(uname -v)" == *"Ubuntu"* ]] ; then
		PKG="Ubuntu"
		echo "Running on Ubuntu"
	else
		echo "Unknown operating system"
		echo "Please select your operating system"
		echo "Choices:"
		echo "	     linux - any linux distro"
		echo "	     darwin - MacOS"
		read PKG
	fi
}

update() {
	echo -e "Start to update && upgrade"
	if [ $PKG = "Ubuntu" ] ; then
                sudo apt update -y && sudo apt upgrade -y
        elif [ $PKG = "Debian" ] ; then
                yum apt update -y && sudo apt upgrade -y
        fi
	echo -e "Update && upgrade finished"
}

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

install_gccgo() {
	echo "Start to install gccgo"
	if [ $PKG = "Ubuntu" ] ; then
		sudo apt-get install gccgo -y
	elif [ $PKG = "Debian" ] ; then
		yum apt-get install gccgo -y
	fi
	gccgo -V
	echo "gccgo installed"
}

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

install_rust() {
	echo "Start to install rust"
	curl https://sh.rustup.rs -sSf | sh -s -- -y
	source $HOME/.cargo/env
	rustup -V
	rustc -V
	cargo -V
	echo "rust installed"
}

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
	echo "openssl installed"
}

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

install_nodejs(){
	echo "Start to install nodejs10"
	curl -sL https://deb.nodesource.com/setup_10.x | bash -
	apt-get install -y nodejs
	node --version
	npm --version
}

install_yarn(){
	echo "Start to install nodejs10"
	curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
	echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
	apt-get update -qq
	apt-get install -y yarn
	yarn --version
}

install_redis(){
	echo "Start to install redis"
	[[ $(uname) = Linux ]] || exit 1
	[[ $USER = root ]] || exit 1

	apt-get --assume-yes install redis
}

check_os
echo "----------------------------------------------------------------------------------------"
update
echo "----------------------------------------------------------------------------------------"
install_python3
echo "----------------------------------------------------------------------------------------"
install_clang
echo "----------------------------------------------------------------------------------------"
install_golang
echo "----------------------------------------------------------------------------------------"
install_rust
echo "----------------------------------------------------------------------------------------"
install_openssl
echo "----------------------------------------------------------------------------------------"
install_influxdb
echo "----------------------------------------------------------------------------------------"
install_pkgconfig
echo "----------------------------------------------------------------------------------------"
install_zlib1g_dev
echo "----------------------------------------------------------------------------------------"
install_nodejs
echo "----------------------------------------------------------------------------------------"
install_yarn
echo "----------------------------------------------------------------------------------------"
install_redis