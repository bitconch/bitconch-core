
#!/usr/bin/env bash
set -ex


install_python_module(){
	echo "Start to install Python Modules"
	[[ $(uname) = Linux ]] || exit 1
	[[ $USER = root ]] || exit 1
	pip3 install colorama
	pip3 install click
}

install_python_module