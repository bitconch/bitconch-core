use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Write;
use std::io::{BufReader, BufWriter};
use std::path::Path;

const CHACHA_IVEC_SIZE: usize = 64;

#[link(name = "cpu-crypt")]
/// set up an integration with the chacha20_cbc_encrypt function from the C standard library
extern "C" {
    fn chacha20_cbc_encrypt(
        input: *const u8,
        output: *mut u8,
        in_len: usize,
        key: *const u8,
        ivec: *mut u8,
    );
}

/// the chachacha_encrypt_str function encapsulates a unsafe block
pub fn chacha_encrypt_str(input: &[u8], output: &mut [u8], key: &[u8], ivec: &mut [u8]) {
/// call the chacha20_cbc_encrypt function within a separate unsafe block
    unsafe {
        chacha20_cbc_encrypt(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
            key.as_ptr(),
            ivec.as_mut_ptr(),
        );
    }
}

/// define the function of chachacha_encrypt_files, an the type of return value is Result<()>
pub fn chacha_encrypt_files(in_path: &Path, out_path: &Path, key: String) -> io::Result<()> {
/// creates a new BufReader with a default buffer capacity and instantiation it 
/// with the parameter of in_path open a file in read-only mode 
/// panics if path does not already exist, with a panic message "Can't open ledger data file"  
    let mut in_file = BufReader::new(File::open(in_path).expect("Can't open ledger data file"));
/// creates a new BufWriter with a default buffer capacity and instantiation it 
/// with the parameter of in_path open a file in read-only mode 
/// if file does not exist then will will create a file, and will truncate it if it does
/// panics if path does not already exist, with a panic message "Can't open ledger encrypted data file"
    let mut out_file =
        BufWriter::new(File::create(out_path).expect("Can't open ledger encrypted data file"));
/// the mutable array named buffer contain 4 * 1024 elements and all be set to the value 0 initially
    let mut buffer = [0; 4 * 1024];
/// the mutable array named encrypted_buffer contain 4 * 1024 elements and all be set to the value 0 initially
    let mut encrypted_buffer = [0; 4 * 1024];
/// the mutable array named ivec contain 64 elements and all be set to the value 0 initially
    let mut ivec = [0; CHACHA_IVEC_SIZE];

/// if referenced mut buffer parameter to read the file successfully
    while let Ok(size) = in_file.read(&mut buffer) {
/// then output of file bytes through macros
        println!("read {} bytes", size);
/// if size == 0, then break
        if size == 0 {
            break;
        }
/// call the function of chacha_encrypt_str
        chacha_encrypt_str(
            &buffer[..size],
            &mut encrypted_buffer[..size],
            key.as_bytes(),
            &mut ivec,
        );
        if let Err(res) = out_file.write(&encrypted_buffer[..size]) {
            println!("Error writing file! {:?}", res);
            return Err(res);
        }
    }
    Ok(())
}

