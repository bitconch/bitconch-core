#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {
        match $expr {
            Ok(expr) => expr,
            Err(err) => {
                crate::errors::ERROR
                    .lock()
                    .replace(failure::Error::from(err));

                return crate::errors::HederaResult::Failure;
            }
        }
    };
}