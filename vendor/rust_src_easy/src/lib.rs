extern crate libc;

#[no_mangle]
pub extern fn hello_world() {
    println!("Hello World!");
}