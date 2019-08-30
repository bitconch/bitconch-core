extern crate bincode;
extern crate bytes;
#[macro_use]
extern crate clap;
extern crate log;
extern crate serde_json;
extern crate buffett_core;
extern crate buffett_metrics;
extern crate buffett_crypto;
extern crate tokio;
extern crate tokio_codec;

use bincode::{deserialize, serialize};
use bytes::Bytes;
use clap::{App, Arg};
use buffett_core::token_service::{Drone, DroneRequest, DRONE_PORT};
use buffett_core::logger;
use buffett_metrics::metrics::set_panic_hook;
use buffett_crypto::signature::read_keypair;
use std::error;
use std::io;
use std::net::{Ipv4Addr, SocketAddr};
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::net::TcpListener;
use tokio::prelude::*;
use tokio_codec::{BytesCodec, Decoder};

macro_rules! socketaddr {
    ($ip:expr, $port:expr) => {
        SocketAddr::from((Ipv4Addr::from($ip), $port))
    };
    ($str:expr) => {{
        let a: SocketAddr = $str.parse().unwrap();
        a
    }};
}

/// declare the function of main
fn main() -> Result<(), Box<error::Error>> {
    /// initialization log
    logger::setup();
    /// if there is panic in "tokenbot" program, then will record the panic information into influxdb database
    set_panic_hook("tokenbot"); 
    /// creates a new instance of an application named "tokenbot"
    /// set the version to the same version of the application as crate automatically at compile time
    /// adds an argument to the list of valid possibilities
    let matches = App::new("tokenbot")
        .version(crate_version!())
        .arg(
            /// creates a new instance of Arg named "network"
            Arg::with_name("network")
                /// sets the short version of the argument
                .short("n")
                /// sets the long version of the argument
                .long("network")
                /// specifies the name for value of option or positional arguments inside of help documentation
                .value_name("HOST:PORT")
                /// specifies that the argument takes a value at run time
                .takes_value(true)
                /// sets whether or not the argument is required by default
                .required(true)
                /// sets the short help text
                .help("Ip and port number of the leader node"),
        ).arg(
            Arg::with_name("keypair")
                .short("k")
                .long("keypair")
                .value_name("PATH")
                .takes_value(true)
                .required(true)
                .help("File from which to read the mint's keypair"),
        ).arg(
            Arg::with_name("slice")
                .long("slice")
                .value_name("SECS")
                .takes_value(true)
                .help("Time interval limit for airdropping request"),
        ).arg(
            Arg::with_name("cap")
                .long("cap")
                .value_name("NUM")
                .takes_value(true)
                .help("Request limit during each interval"),
        /// starts the parsing process, upon a failed parse an error will be displayed to the user 
        /// and the process will exit with the appropriate error code.
        ).get_matches();

    /// gets the value of "network", and parse it,
    /// if faile to parse, then will return the default error message, 
    /// print the error message, and exits the program
    let network = matches
        .value_of("network")
        .unwrap()
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("failed to parse network: {}", e);
            exit(1)
        });

    /// get the keypair from the value of "keypair" in application, 
    /// and print the error message if it fails.
    let mint_keypair =
        read_keypair(matches.value_of("keypair").unwrap()).expect("failed to read client keypair");

    /// declare the "time_slice" variable
    let time_slice: Option<u64>;
    /// destructure the value of "slice" in application
    /// if "slice" wrapped Some value,then parse "slice", and return, if faile to parse then print the error message,
    /// if is None, then return None to time_slice
    if let Some(secs) = matches.value_of("slice") {
        time_slice = Some(secs.to_string().parse().expect("failed to parse slice"));
    } else {
        time_slice = None;
    }
    /// declare the "request_cap" variable
    let request_cap: Option<u64>;
    /// destructure the value of "cap" in application
    /// if "cap" wrapped Some value,then parse "cap", and return, if faile to parse then print the error message,
    /// if is None, then return None to request_cap
    if let Some(c) = matches.value_of("cap") {
        request_cap = Some(c.to_string().parse().expect("failed to parse cap"));
    } else {
        request_cap = None;
    }

    /// get the airdrop address
    let drone_addr = socketaddr!(0, DRONE_PORT);

    /// generate a instance of Drone
    /// using an Arc<T> to wrap the Mutex<T> able to share ownership across multiple threads
    let drone = Arc::new(Mutex::new(Drone::new(
        mint_keypair,
        drone_addr,
        network,
        time_slice,
        request_cap,
    )));

    let drone1 = drone.clone();
    /// create a new thread to loop
    thread::spawn(move || loop {
        /// use "lock" method to get locks to access "time_slice" data in "drone1" mutex
        let time = drone1.lock().unwrap().time_slice;
        /// puts the current thread to sleep
        thread::sleep(time);
        /// obtain locks to access the results of "clear_request_count" function in the "drone1" mutex
        drone1.lock().unwrap().clear_request_count();
    });

    let socket = TcpListener::bind(&drone_addr).unwrap();
    println!("Tokenbot started. Listening on: {}", drone_addr);
    let done = socket
        .incoming()
        .map_err(|e| println!("failed to accept socket; error = {:?}", e))
        .for_each(move |socket| {
            let drone2 = drone.clone();
            // let client_ip = socket.peer_addr().expect("drone peer_addr").ip();
            let framed = BytesCodec::new().framed(socket);
            let (writer, reader) = framed.split();

            let processor = reader.and_then(move |bytes| {
                let req: DroneRequest = deserialize(&bytes).or_else(|err| {
                    Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("deserialize packet in drone: {:?}", err),
                    ))
                })?;

                println!("Airdrop requested...");
                // let res = drone2.lock().unwrap().check_rate_limit(client_ip);
                let res1 = drone2.lock().unwrap().send_airdrop(req);
                match res1 {
                    Ok(_) => println!("Airdrop sent!"),
                    Err(_) => println!("Request limit reached for this time slice"),
                }
                let response = res1?;
                println!("Airdrop tx signature: {:?}", response);
                let response_vec = serialize(&response).or_else(|err| {
                    Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("serialize signature in drone: {:?}", err),
                    ))
                })?;
                let response_bytes = Bytes::from(response_vec.clone());
                Ok(response_bytes)
            });
            let server = writer
                .send_all(processor.or_else(|err| {
                    Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Tokenbot response: {:?}", err),
                    ))
                })).then(|_| Ok(()));
            tokio::spawn(server)
        });
    tokio::run(done);
    Ok(())
}
