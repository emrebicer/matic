#[derive(Debug, PartialEq)]
pub enum Error {
    DatasetTooSmall,
    KMustBePositive,
    InputError(String)
}
