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

fn main() -> () {
    logger::setup();
    set_panic_hook("fullnode");
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

    let (keypair, ncp) = if let Some(i) = matches.value_of("identity") {
        let path = i.to_string();
        if let Ok(file) = File::open(path.clone()) {
            let parse: serde_json::Result<Config> = serde_json::from_reader(file);
            if let Ok(data) = parse {
                (data.keypair(), data.node_info.contact_info.ncp)
            } else {
                eprintln!("failed to parse {}", path);
                exit(1);
            }
        } else {
            eprintln!("failed to read {}", path);
            exit(1);
        }
    } else {
        (Keypair::new(), socketaddr!(0, 8000))
    };

    let ledger_path = matches.value_of("ledger").unwrap();

    
    let network = matches
        .value_of("network")
        .map(|network| network.parse().expect("failed to parse network address"));

    let node = Node::new_with_external_ip(keypair.pubkey(), &ncp);

    
    let node_info = node.info.clone();
    let pubkey = keypair.pubkey();

    let mut fullnode = Fullnode::new(node, ledger_path, keypair, network, false, None);

    
    let leader = match network {
        Some(network) => {
            sample_leader_by_gossip(network, None).expect("can't find leader on network")
        }
        None => node_info,
    };

    let mut client = new_client(&leader);

    
    let drone_addr = match network {
        Some(network) => SocketAddr::new(network.ip(), DRONE_PORT),
        None => SocketAddr::new(ncp.ip(), DRONE_PORT),
    };

    loop {
        let balance = client.sample_balance_by_key(&pubkey).unwrap_or(0);
        info!("Balance value {}", balance);

        if balance >= 50 {
            info!("Enough balance for client");
            break;
        }

        info!("Calling Token-bot for air-drop {}", drone_addr);
        loop {
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

    loop {
        let status = fullnode.handle_role_transition();
        match status {
            Ok(Some(FullnodeReturnType::LeaderRotation)) => (),
            _ => {
                exit(1);
            }
        }
    }
}
