//RusteloResult is a work around for Rust Result data type.

#[repr(u8)]
pub enum RusteloResult {
    Success = 0,
    Failure = 1,
}