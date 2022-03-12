#[derive(Debug, PartialEq)]
pub enum KNNError {
    DatasetTooSmall,
    KMustBePositive
}

#[derive(Debug, PartialEq)]
pub enum LinAlgError {
    DifferentSizedInputs
}
