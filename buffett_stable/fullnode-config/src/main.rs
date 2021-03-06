#[macro_use]
extern crate clap;
extern crate dirs;
extern crate serde_json;
extern crate buffett_core;
extern crate buffett_crypto;

use clap::{App, Arg};
use buffett_core::crdt::FULLNODE_PORT_RANGE;
use buffett_core::fullnode::Config;
use buffett_core::logger;
use buffett_core::netutil::{get_ip_addr, get_public_ip_addr, parse_port_or_addr};
use buffett_crypto::signature::read_pkcs8;
use std::io;
use std::net::SocketAddr;

fn main() {
    /// setting up logs
    logger::setup();
    /// creates a new instance of an application named "fullnode-config"
    /// automatically set the version of the "fullnode-config" application
    /// to the same thing as the crate at compile time througth crate_version! macro
    /// and add arguments to the list of valid possibilities
    /// starts the parsing process, upon a failed parse an error will be displayed to the user 
    /// and the process will exit with the appropriate error code.
    let matches = App::new("fullnode-config")
        .version(crate_version!())
        .arg(
            Arg::with_name("local")
                .short("l")
                .long("local")
                .takes_value(false)
                .help("Detect network address from local machine configuration"),
        ).arg(
            Arg::with_name("keypair")
                .short("k")
                .long("keypair")
                .value_name("PATH")
                .takes_value(true)
                .help("/path/to/id.json"),
        ).arg(
            Arg::with_name("public")
                .short("p")
                .long("public")
                .takes_value(false)
                .help("Detect public network address using public servers"),
        ).arg(
            Arg::with_name("bind")
                .short("b")
                .long("bind")
                .value_name("PORT")
                .takes_value(true)
                .help("Bind to port or address"),
        ).get_matches();


    let bind_addr: SocketAddr = {
        /// generate a instance of SocketAddr 
        /// by the value of command-line parameters "bind" and the first number of "FULLNODE_PORT_RANGE" constants
        let mut bind_addr = parse_port_or_addr(matches.value_of("bind"), FULLNODE_PORT_RANGE.0);
        /// if "local" argument was present at runtime
        /// then get the local IP part of a given IpNetwork, and set the ip of "bind_addr"
        if matches.is_present("local") {
            let ip = get_ip_addr().unwrap();
            bind_addr.set_ip(ip);
        }
        /// if "public" argument was present at runtime
        /// then get the public IP part of a given IpNetwork, and set the ip of "bind_addr"
        if matches.is_present("public") {
            let ip = get_public_ip_addr().unwrap();
            bind_addr.set_ip(ip);
        }
        /// return the value of "bind_addr"
        bind_addr
    };

    /// get the path to the user's home directory
    let mut path = dirs::home_dir().expect("home directory");
    /// if "keypair" argument was present at runtime, then get the value of "keypair"
    /// if not existent, then Extends the "path" with the contents of an iterato, 
    /// and turn then "path" into a &str slice, and return it
    let id_path = if matches.is_present("keypair") {
        matches.value_of("keypair").unwrap()
    } else {
        path.extend(&[".config", "bitconch", "id.json"]);
        path.to_str().unwrap()
    };
    /// generate pkcs8 vector by "id_path", if failed then call panicand print the error information
    let pkcs8 = read_pkcs8(id_path).expect("client keypair");

    
    /// generate a Config instance
    let config = Config::new(&bind_addr, pkcs8);
    /// constructs a new handle to the standard output of the current process
    let stdout = io::stdout();
    /// serialize data structure of "stdout, &config" as JSON into the IO stream
    /// if failed then call panicand print the error information
    serde_json::to_writer(stdout, &config).expect("serialize");
}
