extern crate atty;
#[macro_use]
extern crate clap;
extern crate serde_json;
extern crate buffett_core;

use atty::{is, Stream};
use clap::{App, Arg};
use buffett_core::ledger::LedgerWriter;
use buffett_core::coinery::Mint;
use std::error;
use std::io::{stdin, Read};
use std::process::exit;

/// declare main function, return value type is Result <T, E>, where T is (), E is Box <error:: Error>
fn main() -> Result<(), Box<error::Error>> {
    /// creates a new instance of an application named "bitconch-genesis"
    /// automatically set the version of the "bitconch-genesis" application
    /// to the same thing as the crate at compile time througth crate_version! macro
    /// and add arguments to the list of valid possibilities
    /// starts the parsing process, upon a failed parse an error will be displayed to the user 
    /// and the process will exit with the appropriate error code
    let matches = App::new("bitconch-genesis")
        .version(crate_version!())
        .arg(
            Arg::with_name("tokens")
                .short("t")
                .long("tokens")
                .value_name("NUM")
                .takes_value(true)
                .required(true)
                .help("Number of tokens with which to initialize mint"),
        ).arg(
            Arg::with_name("ledger")
                .short("l")
                .long("ledger")
                .value_name("DIR")
                .takes_value(true)
                .required(true)
                .help("Use directory as a dedicated ledgerbook path"),
        ).get_matches();

    let tokens = value_t_or_exit!(matches, "tokens", i64);
    let ledger_path = matches.value_of("ledger").unwrap();

    if is(Stream::Stdin) {
        eprintln!("nothing found on stdin, expected a json file");
        exit(1);
    }

    let mut buffer = String::new();
    let num_bytes = stdin().read_to_string(&mut buffer)?;
    if num_bytes == 0 {
        eprintln!("empty file on stdin, expected a json file");
        exit(1);
    }

    let pkcs8: Vec<u8> = serde_json::from_str(&buffer)?;
    let mint = Mint::new_with_pkcs8(tokens, pkcs8);

    let mut ledger_writer = LedgerWriter::open(&ledger_path, true)?;
    ledger_writer.write_entries(mint.create_entries())?;

    Ok(())
}
