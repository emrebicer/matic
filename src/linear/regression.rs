use crate::error::Error;
use crate::math::linalg::dot;

pub struct LinearRegression {
    // Weights
    coefficients: Vec<f64>,
    // Bias
    intercept: f64,
    max_iter: usize,
}

impl LinearRegression {
    /// Linear regression; calculates the coefficients and intercept that fits the data
    ///
    ///
    /// # Parameters
    ///
    /// `training_data` is a vector that contains tuples
    ///
    /// `training_data.0` is a vector that contains training inputs,
    ///     every vector in it should have the same length
    ///
    /// `training_data.1` is a vector that contains target values
    ///
    /// # Returns
    ///
    /// A tuple of 2 values; first value is coefficients (a vector of weights)
    /// and the second valueis the intercept (bias)
    ///
    pub fn fit(&self, training_data: &Vec<(Vec<f64>, f64)>) -> Result<(Vec<f64>, f64), Error> {
        if training_data.is_empty() {
            return Err(Error::InputError("training_data can't be empty"));
        }

        // Make sure the training data contains the same sized vectors
        let feature_size = training_data.first().unwrap().0.len();

        for vals in training_data.iter() {
            if vals.0.len() != feature_size {
                return Err(Error::InputError(
                    "training_data can't contain different sized vectors",
                ));
            }
        }

        let mut iteration_count = 0;

        // Initialize coefficients
        let mut coefficients = vec![];
        let mut intercept = 1;

        for _ in 0..feature_size {
            coefficients.push(1);
        }

        while iteration_count < self.max_iter {
            iteration_count += 1;
        }

        unimplemented!();
    }

    /// Calculates the prediction given weight and input
    /// prediction = f(x) = ax + b
    pub fn predict(&self, x: &Vec<f64>) -> Result<f64, Error> {
        return match dot(x, &self.coefficients) {
            Ok(val) => Ok(val + self.intercept),
            Err(err) => Err(err),
        };
    }
}
