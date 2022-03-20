#[derive(Debug, PartialEq)]
pub enum Error {
    DatasetTooSmall,
    KMustBePositive,
    NonInvertibleMatrix,
    InputError(String)
}
