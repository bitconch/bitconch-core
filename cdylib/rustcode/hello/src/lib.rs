extern crate libc;
//#[macro_use]
//extern crate clap;

mod rustelo_error;
#[macro_use]
mod macros;
mod a_simple_proc;
mod resutl;


use std::ffi::CStr;
use std::ffi::c_void;
use std::error;
//use clap::{App, Arg};
use rustelo_error::RusteloResult;

#[no_mangle]
pub extern "C" fn hello(param1: *const libc::c_char,
                        param2: *const libc::c_char)->RusteloResult {
    /*
    let buf_param1 = unsafe { CStr::from_ptr(param1).to_bytes() };
    let str_param1 = String::from_utf8(buf_param1.to_vec()).unwrap();
    println!("Hello {}!", str_param1);
    let buf_param2 = unsafe { CStr::from_ptr(param2).to_bytes() };
    let str_param2 = String::from_utf8(buf_param2.to_vec()).unwrap();
    println!("Hello {}!", str_param2);
    */

    RusteloResult::Success
}

 
//use xi4win::main_entry::main_entry_point;



#[no_mangle]
pub extern "C" fn rustcode_clap_cli(network: *const libc::c_char,
                                    identity:*const libc::c_char,
                                    threshold:*const libc::c_char) -> Result<(), Box<std::error::Error>>{
/*
pub extern "C" fn rustcode_clap_cli(network: *const libc::c_char,
                                    identity:*const libc::c_char,
                                    threshold:*const libc::c_char) -> RusteloResult{
*/
   
    println!("This is a simple CLI created by Clap");

    let mut path = dirs::home_dir().expect("home directory");
    println!("Marker 1");
    let network_str_ =  unsafe { CStr::from_ptr(network) .to_str().unwrap()};
    println!("Marker 2");
    let identity_str_ =  unsafe { CStr::from_ptr(identity) }.to_str().unwrap();
    println!("Marker 3");
    let threshold_str_ =  unsafe { CStr::from_ptr(threshold) }.to_str().unwrap();
    println!("Marker 4");
     
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
    
    //RusteloResult::Success
    Ok(())
}

/*
#[no_mangle]
pub extern "C" fn hello_dummy(){
    println!("Hello Dummy");
}
*/

/*
//define a struct for argument 
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct CArgument {
    network:   *const libc::c_char,
    identity:  *const libc::c_char,
    threshold: *const libc::c_char,

}

pub struct S010arg01 {
    network:   *const libc::c_char,
    identity:  *const libc::c_char,
    threshold: *const libc::c_char,
}

impl S010arg01 {
    pub fn new() -> Result<Self, Error> {

    
    Ok(Self {
            network,
            identity,
            threshold,
        })
    }
}
*/
/* 
#[no_mangle]
pub extern "C" fn rustelo_s010arg01_new(out: *mut *mut S010arg01) -> RusteloResult {
    
    let s010arg01 = try_ffi!(S010arg01::new());

        unsafe {
            *out = Box::into_raw(Box::new(s010arg01));
        }

    RusteloResult::Success
}
*/

/*   
#[no_mangle]
pub unsafe extern "C" fn rustcode_clap_cli_struct(args : *mut CArgument) -> RusteloResult{  
    println!("This is a simple CLI created by Clap");
    println!("The network is {:?}", (&mut *args).network);
    println!("The network is {:?}", (&mut *args).identity);
    println!("The network is {:?}", (&mut *args).threshold);
    RusteloResult::Success
}
*/