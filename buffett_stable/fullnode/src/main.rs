#[macro_use]
extern crate clap;
extern crate getopts;
#[macro_use]
extern crate log;
extern crate serde_json;
#[macro_use]
extern crate buffett_core;
extern crate buffett_metrics;
extern crate buffett_crypto;

use clap::{App, Arg};
use buffett_core::client::new_client;
use buffett_core::crdt::Node;
use buffett_core::token_service::DRONE_PORT;
use buffett_core::fullnode::{Config, Fullnode, FullnodeReturnType};
use buffett_core::logger;
use buffett_metrics::metrics::set_panic_hook;
use buffett_crypto::signature::{Keypair,KeypairUtil};
use buffett_core::thin_client::sample_leader_by_gossip;
use buffett_core::wallet::request_airdrop;
use std::fs::File;
use std::net::{Ipv4Addr, SocketAddr};
use std::process::exit;
use std::thread::sleep;
use std::time::Duration;

/// declares a main function
fn main() -> () {
    /// setting up logs
    logger::setup();
    /// if there is panic in "fullnode" program, then will record the panic information into influxdb database
    set_panic_hook("fullnode");
    /// creates a new instance of an application named "fullnode"
    /// automatically set the version of the "fullnode" application
    /// to the same thing as the crate at compile time througth crate_version! macro
    /// and add arguments to the list of valid possibilities
    /// starts the parsing process, upon a failed parse an error will be displayed to the user 
    /// and the process will exit with the appropriate error code. 
    let matches = App::new("fullnode")
        .version(crate_version!())
        .arg(
            Arg::with_name("identity")
                .short("i")
                .long("identity")
                .value_name("PATH")
                .takes_value(true)
                .help("Run with the identity found in FILE"),
        ).arg(
            Arg::with_name("network")
                .short("n")
                .long("network")
                .value_name("HOST:PORT")
                .takes_value(true)
                .help("Rendezvous with the network at this gossip entry point"),
        ).arg(
            Arg::with_name("ledger")
                .short("l")
                .long("ledger")
                .value_name("DIR")
                .takes_value(true)
                .required(true)
                .help("use DIR as a dedicated ledgerbook path"),
        ).get_matches();

    /// destructures "matches" into "Some(i)", get the value of "identity" successfully
    let (keypair, ncp) = if let Some(i) = matches.value_of("identity") {
        /// Converts "i" to a string
        let path = i.to_string();
        /// if open the file in read-only mode successfully whit "path"
        if let Ok(file) = File::open(path.clone()) {
            /// deserialize an instance "file" from an IO stream of JSON
            let parse: serde_json::Result<Config> = serde_json::from_reader(file);
            /// if destructures "parse" into "Ok(data)"
            /// then return the "keypair" and "ncp" of the file
            if let Ok(data) = parse {
                (data.keypair(), data.node_info.contact_info.ncp)
                /// if fail to destructures
                /// then print the error message and exit the program
            } else {
                eprintln!("failed to parse {}", path);
                exit(1);
            }
            /// if fail to destructures "parse"
            /// then print the error message and exit the program
        } else {
            eprintln!("failed to read {}", path);
            exit(1);
        }
        /// if fail to open the file, evaluate the block "{}"
    } else {
        (Keypair::new(), socketaddr!(0, 8000))
    };

    /// gets the value of "ledger" argument
    let ledger_path = matches.value_of("ledger").unwrap();

    
    /// gets the value of "network" argument
    /// takes a closure and creates an iterato to parse network address
    let network = matches
        .value_of("network")
        .map(|network| network.parse().expect("failed to parse network address"));

    /// generate a "Node" instance
    let node = Node::new_with_external_ip(keypair.pubkey(), &ncp);

    
    /// clnoe "info's NodeInfo" from "node"
    let node_info = node.info.clone();
    let pubkey = keypair.pubkey();

    /// new a "Fullnode" instance
    let mut fullnode = Fullnode::new(node, ledger_path, keypair, network, false, None);

    
    /// match with "network"
    /// if "network" in a Some value, generate a OK value of "NodeInfo"
    /// if it is false then call panic! and print the error message
    let leader = match network {
        Some(network) => {
            sample_leader_by_gossip(network, None).expect("can't find leader on network")
        }
        /// if is None, then return "node_info"
        None => node_info,
    };

    /// reference to "leader" to create a new client
    let mut client = new_client(&leader);

    
    /// match with "network"
    ///  if "network" in a Some value, new a SocketAddr instance with "network.ip(), DRONE_PORT"
    /// if is None, new a SocketAddr instance with "ncp.ip(), DRONE_PORT"
    let drone_addr = match network {
        Some(network) => SocketAddr::new(network.ip(), DRONE_PORT),
        None => SocketAddr::new(ncp.ip(), DRONE_PORT),
    };

    /// start loop
    loop {
        /// get the balance of client by referring to the pubkey every 100 milliseconds, 
        /// and write the consumed time and balance to influxdb database.
        /// If it exceeds 1 second (timeout), return "0" and write the consumed time into influxdbdatabase
        let balance = client.sample_balance_by_key(&pubkey).unwrap_or(0);
        /// logging "balance" at the info log level
        info!("Balance value {}", balance);

        if balance >= 50 {
            info!("Enough balance for client");
            break;
        }

        info!("Calling Token-bot for air-drop {}", drone_addr);
        /// loop
        loop {
            /// if the rusult of signature is ture, quit the loop 
            if request_airdrop(&drone_addr, &pubkey, 50).is_ok() {
                break;
            }
            info!(
                "Token-bot on {:?} status unkown",
                drone_addr
            );
            sleep(Duration::from_secs(2));
        }
    }

    /// loop
    loop {
        /// get the result of "FullnodeReturnType::LeaderRotation"
        let status = fullnode.handle_role_transition();
        /// if the result of "Some(FullnodeReturnType::LeaderRotation)" in a OK value,then return "()"
        /// otherwise exit the program
        match status {
            Ok(Some(FullnodeReturnType::LeaderRotation)) => (),
            _ => {
                exit(1);
            }
        }
    }
}
