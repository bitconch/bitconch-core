extern crate libc;
#[macro_use]
extern crate clap;

use std::ffi::CStr;
use std::ffi::c_void;
use std::error;
use clap::{App, Arg};



#[no_mangle]
pub extern "C" fn hello(name: *const libc::c_char) {
    let buf_name = unsafe { CStr::from_ptr(name).to_bytes() };
    let str_name = String::from_utf8(buf_name.to_vec()).unwrap();
    println!("Hello {}!", str_name);
}

 
//use xi4win::main_entry::main_entry_point;



#[no_mangle]
pub extern "C" fn rustcode_clap_cli() -> Result<(), Box<error::Error>>{
    
   println!("This is a simple CLI created by Clap");
   //main_entry_point();
    let matches = App::new("simple-clap-cli")
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
        println!("argument outfile is present");
        matches.value_of("outfile").unwrap()
    } else {
        println!("argument outfile is NOT present");
        path.extend(&[".config", "solana", "id.json"]);
        path.to_str().unwrap()
    };
    println!("This is the PATH value {}",outfile);
    //let serialized_keypair = gen_keypair_file(outfile.to_string())?;
    /*
    if outfile == "-" {
        println!("{}", serialized_keypair);
    }
    */
    Ok(())
}
/*
#[no_mangle]
pub extern "C" fn hello_dummy(){
    println!("Hello Dummy");
}
*/