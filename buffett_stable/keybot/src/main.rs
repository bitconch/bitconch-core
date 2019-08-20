#[macro_use]
extern crate clap;
extern crate dirs;
extern crate serde_json;
extern crate buffett_core;

use clap::{App, Arg};
use buffett_core::wallet::gen_keypair_file;
use std::error;

/// declare main function, return value type is Result <T, E>, where T is (), E is Box <error:: Error>
fn main() -> Result<(), Box<error::Error>> {
    /// creates a new instance of an application named "buffettt_tokenbot",
    /// automatically set the version of the "buffettt_tokenbot" application,
    /// starts the parsing process, upon a failed parse an error will be displayed to the user 
    /// and the process will exit with the appropriate error code
    let matches = App::new("buffettt_tokenbot")
        .version(crate_version!())
        .arg(
            Arg::with_name("outfile")
                .short("o")
                .long("outfile")
                .value_name("PATH")
                .takes_value(true)
                .help("Path to generated file"),
        ).get_matches();

    /// get the path to the user's home directory
    let mut path = dirs::home_dir().expect("home directory");
    /// if argument of "outfile" was present at runtime,
    /// then get the value of "outfile"
    /// otherwise extends the "path" with [".config", "bitconch", "id.json"],and return a &str slice path 
    let outfile = if matches.is_present("outfile") {
        matches.value_of("outfile").unwrap()
    } else {
        path.extend(&[".config", "bitconch", "id.json"]);
        path.to_str().unwrap()
    };

    /// converts the str of "outfile" to string
    let _tmp = outfile.to_string();
    /// serialize the PKCS#8 document of the new key pair as a String of JSON.
    let serialized_keypair = gen_keypair_file(_tmp)?;
    if outfile == "-" {
        println!("{}", serialized_keypair);
    }
    /// return the success value
    Ok(())
}
