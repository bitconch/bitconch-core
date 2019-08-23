#[macro_use]
extern crate clap;
extern crate serde_json;
extern crate buffett_core;

use clap::{App, Arg, SubCommand};
use buffett_core::tx_vault::Bank;
use buffett_core::ledger::{read_ledger, verify_ledger};
use buffett_core::logger;
use std::io::{stdout, Write};
use std::process::exit;

fn main() {
    /// setting up logs
    logger::setup();
    /// creates a new instance of an application named "ledger-tool",
    /// automatically set the version of the "ledger-tool" application,
    /// starts the parsing process, upon a failed parse an error will be displayed to the user 
    /// and the process will exit with the appropriate error code
    let matches = App::new("ledger-tool")
        .version(crate_version!())
        .arg(
            Arg::with_name("ledger")
                .short("l")
                .long("ledger")
                .value_name("DIR")
                .takes_value(true)
                .required(true)
                .help("Use directory for ledger location"),
        )
        .arg(
            Arg::with_name("head")
                .short("n")
                .long("head")
                .value_name("NUM")
                .takes_value(true)
                .help("Specify the number of the most recent entries in the ledgerbook\n  (only applies to verify, print, json commands)"),
        )
        .arg(
            Arg::with_name("precheck")
                .short("p")
                .long("precheck")
                .help("Use ledger_verify() to check internal ledger consistency before proceeding"),
        )
        .arg(
            Arg::with_name("continue")
                .short("c")
                .long("continue")
                .help("Continue verify even if verification fails"),
        )
        /// creates a new instance of a subcommand named "print",
        /// sets a string with "Print the ledger" describing what the program does,
        /// and brings both pieces of information together
        .subcommand(SubCommand::with_name("print").about("Print the ledger"))
        .subcommand(SubCommand::with_name("json").about("Print the ledger in JSON format"))
        .subcommand(SubCommand::with_name("verify").about("Verify the ledger's PoH"))
        .get_matches();

    /// get information about the "ledger"
    let ledger_path = matches.value_of("ledger").unwrap();

    /// if argument of "precheck" was present at runtime
    /// reference to "ledger_path" to verify the ledger , 
    /// if failed, print the error information and exit the program 
    if matches.is_present("precheck") {
        if let Err(e) = verify_ledger(&ledger_path) {
            eprintln!("ledger precheck failed, error: {:?} ", e);
            exit(1);
        }
    }

    /// destructure the iterator of all the entries in "ledger_path"
    /// if in a Ok value, then return "entries"
    /// if is error, then print the error message and exit the program
    let entries = match read_ledger(ledger_path, true) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("Failed to open ledger at {}: {}", ledger_path, err);
            exit(1);
        }
    };

    /// destructure the value of "head"
    /// if "head" in Some value, then parse head，call panic! when it failed, and print the error message
    /// if is None,then Returns the largest value that can be represented by usize type
    let head = match matches.value_of("head") {
        Some(head) => head.parse().expect("please pass a number for --head"),
        None => <usize>::max_value(),
    };

    /// destructure the SubCommand
    match matches.subcommand() {
        /// if the subcommand is ("print", _), then destructure the iterator of all the entries in "ledger_path"
        /// if in a Ok value, then return "entries"
        /// if is error, then print the error message and exit the program
        ("print", _) => {
            let entries = match read_ledger(ledger_path, true) {
                Ok(entries) => entries,
                Err(err) => {
                    eprintln!("Failed to open ledger at {}: {}", ledger_path, err);
                    exit(1);
                }
            };
            /// creates an iterator which gives the current iteration count and the next value
            /// where "i" is the current index of iteration and "entry "is the value returned by the iterator
            /// if i >= head, then quit the loop
            for (i, entry) in entries.enumerate() {
                if i >= head {
                    break;
                }
                let entry = entry.unwrap();
                println!("{:?}", entry);
            }
        }
        /// if the subcommand is ("json", _),
        /// write an entire buffer of " b"{\"ledger\":[\n" " into stdout(), if failed, call panic! and print the error message
        ("json", _) => {
            stdout().write_all(b"{\"ledger\":[\n").expect("open array");
            /// creates an iterator which gives the current iteration count and the next value
            /// if i >= head, then quit the loop
            for (i, entry) in entries.enumerate() {
                if i >= head {
                    break;
                }
                let entry = entry.unwrap();
                /// serialize the standard output and entry as JSON into the IO stream
                serde_json::to_writer(stdout(), &entry).expect("serialize");
                /// write an entire buffer of " b"{\"ledger\":[\n" " the standard output, if failed, call panic! and print the error message
                stdout().write_all(b",\n").expect("newline");
            }
            stdout().write_all(b"\n]}\n").expect("close array");
        }
        /// if the subcommand is ("verify", _), then evaluate the block "{}"
        /// if head < 2,print the error message and exit
        ("verify", _) => {
            if head < 2 {
                eprintln!("verify requires at least 2 entries to run");
                exit(1);
            }
            /// generate a instances of Bank
            let bank = Bank::default();

            {
                /// destructures the iterator of all the entries in "ledger_path"
                /// if in a Ok value, then return "entries"
                /// if is error, then print the error message and exit the program
                let genesis = match read_ledger(ledger_path, true) {
                    Ok(entries) => entries,
                    Err(err) => {
                        eprintln!("Failed to open ledger at {}: {}", ledger_path, err);
                        exit(1);
                    }
                };

                /// takes a closure and creates an iterator that yields its first 2 elements 
                /// and calls the closure on each element
                let genesis = genesis.take(2).map(|e| e.unwrap());

                /// destructure processes bank accounts by "genesis" to return the total entry of the account, 
                /// and if it fails, print the error information.
                /// If "continue" of the command parameter of matches does not present at running time, 
                /// then exit the program
                if let Err(e) = bank.process_ledger(genesis) {
                    eprintln!("verify failed at genesis err: {:?}", e);
                    if !matches.is_present("continue") {
                        exit(1);
                    }
                }
            }
            /// takes a closure and creates an iterator which calls that closure on each element
            let entries = entries.map(|e| e.unwrap());

            let head = head - 2;
            /// creates an iterator that skips the first 2 elements 
            /// and gives the current iteration count and the next value
            /// if i >= head, then quit the loop
            for (i, entry) in entries.skip(2).enumerate() {
                if i >= head {
                    break;
                }
                /// if reference the last id hash value of the bank to verifies the hashes and counts 
                /// of a slice of transactions are inconsistent, then print the error message.
                /// If "continue" of the command parameter of matches does not present at running time, 
                /// then exit the program 
                if !entry.verify(&bank.last_id()) {
                    eprintln!("entry.verify() failed at entry[{}]", i + 2);
                    if !matches.is_present("continue") {
                        exit(1);
                    }
                }
                if let Err(e) = bank.process_entry(&entry) {
                    eprintln!("verify failed at entry[{}], err: {:?}", i + 2, e);
                    if !matches.is_present("continue") {
                        exit(1);
                    }
                }
            }
        }
        ("", _) => {
            eprintln!("{}", matches.usage());
            exit(1);
        }
        _ => unreachable!(),
    };
}
