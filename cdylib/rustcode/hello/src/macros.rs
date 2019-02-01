#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {
        match $expr {
            Ok(expr) => expr,
            Err(err) => {
                crate::rustelo_error::ERROR
                    .lock()
                    .replace(failure::Error::from(err));

                return crate::rustelo_error::RusteloResult::Failure;
            }
        }
    };
}