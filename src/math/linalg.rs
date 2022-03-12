use crate::error::LinAlgError;
use std::iter::zip;

pub fn dot(a: Vec<f64>, b: Vec<f64>) -> Result<f64, LinAlgError> {
    if a.len() != b.len() {
        return Err(LinAlgError::DifferentSizedInputs);
    }

    return Ok(zip(a, b)
        .map(|val| val.0 * val.1)
        .fold(0., |acc, x| acc + x));
}

#[cfg(test)]
mod tests {
    use super::dot;
    use crate::error::LinAlgError;

    #[test]
    fn dot_test() {
        assert_eq!(dot(vec![1., 3., -5.], vec![4., -2., -1.]), Ok(3.));
        assert_eq!(
            dot(vec![1., 3.], vec![4., -2., -1.]),
            Err(LinAlgError::DifferentSizedInputs)
        );
        assert_eq!(
            dot(vec![1., 3.,-5.], vec![4., -2.]),
            Err(LinAlgError::DifferentSizedInputs)
        );
        assert_eq!(dot(vec![], vec![]), Ok(0.));
    }
}
