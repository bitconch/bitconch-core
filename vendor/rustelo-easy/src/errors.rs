use failure::Error;
use libc::c_char;
use mbox::MString;
use once_cell::{sync::Lazy, sync_lazy};
use parking_lot::Mutex;
use std::ptr::null_mut;

//RusteloResult is a workaround for Rust Result data type.
#[repr(u8)]
pub enum RusteloResult {
    Success = 0,
    Failure = 1,
}

//rustelo_handle_error returns an error 
#[no_mangle]
pub extern "C" fn rustelo_handle_error() -> *mut c_char {
    match ERROR.lock().take() {
        Some(err) => MString::from_str(&err.to_string())
            .into_mbox_with_sentinel()
            .into_raw() as _,

        None => null_mut(),
    }
}