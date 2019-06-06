use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Write;
use std::io::{BufReader, BufWriter};
use std::path::Path;

const CHACHA_IVEC_SIZE: usize = 64;

#[link(name = "cpu-crypt")]
extern "C" {
    fn chacha20_cbc_encrypt(
        input: *const u8,
        output: *mut u8,
        in_len: usize,
        key: *const u8,
        ivec: *mut u8,
    );
}

pub fn chacha_encrypt_str(input: &[u8], output: &mut [u8], key: &[u8], ivec: &mut [u8]) {
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

pub fn chacha_encrypt_files(in_path: &Path, out_path: &Path, key: String) -> io::Result<()> {
    let mut in_file = BufReader::new(File::open(in_path).expect("Can't open ledger data file"));
    let mut out_file =
        BufWriter::new(File::create(out_path).expect("Can't open ledger encrypted data file"));
    let mut buffer = [0; 4 * 1024];
    let mut encrypted_buffer = [0; 4 * 1024];
    let mut ivec = [0; CHACHA_IVEC_SIZE];

    while let Ok(size) = in_file.read(&mut buffer) {
        println!("read {} bytes", size);
        if size == 0 {
            break;
        }
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

