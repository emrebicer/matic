use crate::math::linalg::dot;
use crate::error::LinAlgError;

/// Linear regression; calculates the weights and bias that fits the data
/// 
/// 
/// # Parameters
///
/// `x` is a vector that contains training data, every set in the x should have
/// the same length
/// `y` is a vector that contains target values
///
/// # Returns
///
/// A tuple of 2 values; first value is a list of weights and the second value
/// is the bias
///
pub fn fit(x: Vec<Vec<f64>>, y: Vec<f64>) -> (Vec<f64>, f64) {


    unimplemented!();
}


/// Calculates the prediction given weight and input
/// prediction = f(x) = ax + b
pub fn predict(x: Vec<f64>, a: Vec<f64>, b: f64) -> Result<f64, LinAlgError> {
    return match dot(x, a) {
        Ok(val) => Ok(val + b),
        Err(err) => Err(err)
    };
}
