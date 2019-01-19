/*
#[cfg(test)]
mod tests {
    use chacha::chacha_cbc_encrypt_files;
    use std::fs::remove_file;
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_encrypt_file() {
        let in_path = Path::new("test_chacha_encrypt_file_input.txt");
        let out_path = Path::new("test_chacha_encrypt_file_output.txt.enc");
        {
            let mut in_file = File::create(in_path).unwrap();
            in_file.write("123456foobar".as_bytes()).unwrap();
        }
        assert!(chacha_cbc_encrypt_files(in_path, out_path, "thetestkey".to_string()).is_ok());
        let mut out_file = File::open(out_path).unwrap();
        let mut buf = vec![];
        let size = out_file.read_to_end(&mut buf).unwrap();
        assert_eq!(
            buf[..size],
            [106, 186, 59, 108, 165, 33, 118, 212, 70, 238, 205, 185]
        );
        remove_file(in_path).unwrap();
        remove_file(out_path).unwrap();
    }
}
*/
