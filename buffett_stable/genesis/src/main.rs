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

    /// getting a typed value of i64
    let tokens = value_t_or_exit!(matches, "tokens", i64);
    /// get the value of "ledger"
    let ledger_path = matches.value_of("ledger").unwrap();

    /// if Stream enumeration variants of Stdin is tty, then output the error message and exit the program
    if is(Stream::Stdin) {
        eprintln!("nothing found on stdin, expected a json file");
        exit(1);
    }

    /// new a String
    let mut buffer = String::new();
    /// reference to "mut buffer" to read all bytes until EOF in this source, appending them to buf
    let num_bytes = stdin().read_to_string(&mut buffer)?;
    if num_bytes == 0 {
        eprintln!("empty file on stdin, expected a json file");
        exit(1);
    }

    /// parse the string of "buffer" into vector
    let pkcs8: Vec<u8> = serde_json::from_str(&buffer)?;
    /// generation instances of Mint by through tokens and pkcs8
    let mint = Mint::new_with_pkcs8(tokens, pkcs8);

    /// opens or creates a LedgerWriter in ledger_path directory
    let mut ledger_writer = LedgerWriter::open(&ledger_path, true)?;
    ledger_writer.write_entries(mint.create_entries())?;

    Ok(())
}
