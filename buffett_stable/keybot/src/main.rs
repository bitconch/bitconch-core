#[macro_use]
extern crate clap;
extern crate dirs;
extern crate serde_json;
extern crate buffett_core;

use clap::{App, Arg};
use buffett_core::wallet::gen_keypair_file;
use std::error;

fn main() -> Result<(), Box<error::Error>> {
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

    let mut path = dirs::home_dir().expect("home directory");
    let outfile = if matches.is_present("outfile") {
        matches.value_of("outfile").unwrap()
    } else {
        path.extend(&[".config", "bitconch", "id.json"]);
        path.to_str().unwrap()
    };

    let _tmp = outfile.to_string();
    let serialized_keypair = gen_keypair_file(_tmp)?;
    if outfile == "-" {
        println!("{}", serialized_keypair);
    }
    Ok(())
}
