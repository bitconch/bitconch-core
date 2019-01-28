extern crate libc;
#[macro_use]
extern crate clap;

use std::ffi::CStr;
use std::ffi::c_void;
use std::error;
use clap::{App, Arg};

#[repr(u8)]
pub enum RusteloResult {
    Success = 0,
    Failure = 1,
}

#[no_mangle]
pub extern "C" fn hello(param1: *const libc::c_char,
                        param2: *const libc::c_char)->RusteloResult {
    let buf_param1 = unsafe { CStr::from_ptr(param1).to_bytes() };
    let str_param1 = String::from_utf8(buf_param1.to_vec()).unwrap();
    println!("Hello {}!", str_param1);
    let buf_param2 = unsafe { CStr::from_ptr(param2).to_bytes() };
    let str_param2 = String::from_utf8(buf_param2.to_vec()).unwrap();
    println!("Hello {}!", str_param2);
    

    RusteloResult::Success
}

 
//use xi4win::main_entry::main_entry_point;



#[no_mangle]
pub extern "C" fn rustcode_clap_cli(network: *const libc::c_char,
                                    identity:*const libc::c_char,
                                    threshold:*const libc::c_char) -> RusteloResult{

   
   println!("This is a simple CLI created by Clap");
   /* 
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
    */

    let mut path = dirs::home_dir().expect("home directory");
    /*
    let outfile_buff = unsafe  {std::ffi::CStr::from_ptr(outfile_arg).to_bytes()};
    let outfile_str = String::from_utf8(outfile_buff.to_vec()).unwrap();

    let outfile_string_lossy =  unsafe { CStr::from_ptr(outfile_arg) }.to_string_lossy();
    */

    let network_str_ =  unsafe { CStr::from_ptr(network) }.to_str().unwrap();
    let identity_str_ =  unsafe { CStr::from_ptr(identity) }.to_str().unwrap();
    let threshold_str_ =  unsafe { CStr::from_ptr(threshold) }.to_str().unwrap();
    //println!(outfile);
    /*
    let outfile = if matches.is_present("outfile") {
        println!("argument outfile is present");
        matches.value_of("outfile").unwrap()
    } else {
        println!("argument outfile is NOT present");
        path.extend(&[".config", "solana", "id.json"]);
        path.to_str().unwrap()
    };
    */
     
    // handle network 
    let network_id = if !network_str_.is_empty() {
        println!("argument for network is present, using the value from argument");
        //outfile_string_lossy
        //String::from(outfile_str)
        network_str_
    }else{
        println!("argument for networks is NOT present, using default value");
        //path.extend(&[".config", "solana", "id.json"]);
        //path.to_str().unwrap()
        "127.0.0.1:8001"
    };
    println!("The network value is {:?}",network_id.to_string());

    // handle identity
    let identity_path = if !identity_str_.is_empty() {
        println!("argument for identity is present, using the value from argument");
        identity_str_
    }else{
        println!("argument for identity is NOT present, using default value");
        "user/bin/local/default_file_location"
    };
    println!("The identity path is {:?}",identity_path.to_string());

    // handle threshold num
    let threshold_nums = if !threshold_str_.is_empty() {
        println!("argument for threshold is present, using the value from argument");
        threshold_str_
    }else{
        println!("argument for threshold is NOT present, using default value");
        "80808"
    };
    println!("The threshold value is {:?}",threshold_nums.to_string());
    
    
    
    /*
    println!("This is the PATH value {:?}",outfile_str_loss);
    println!("This is the PATH value {:?}",outfile_str_);
    assert_eq!(outfile_str.is_empty(), true);
    assert_eq!(outfile_str_loss.is_empty(), true);
    */
    //assert_eq!(outfile_str_.is_empty(), true);
    //let serialized_keypair = gen_keypair_file(outfile.to_string())?;
    /*
    if outfile == "-" {
        println!("{}", serialized_keypair);
    }
    */
    RusteloResult::Success
}
/*
#[no_mangle]
pub extern "C" fn hello_dummy(){
    println!("Hello Dummy");
}
*/