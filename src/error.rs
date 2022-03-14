#[derive(Debug, PartialEq)]
pub enum Error {
    DatasetTooSmall,
    KMustBePositive,
    InputError(& 'static str)
}
