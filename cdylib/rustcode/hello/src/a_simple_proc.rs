use std::ffi::CStr;
use std::ffi::c_void;
use std::error;
//use clap::{App, Arg};
use rustelo_error::RusteloResult;

#[no_mangle]
pub extern "C" fn simple_proc(param1: *const libc::c_char,
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

