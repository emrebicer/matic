use crate::error::Error;
use crate::math::linalg::Matrix2d;

pub struct LinearRegression {
    // Weights
    coefficients: Matrix2d
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coefficients: Matrix2d::new((1, 1)).unwrap()
        }
    }
    /// Linear regression; calculates the coefficients and intercept that fits
    /// the given training data
    ///
    /// # Parameters
    ///
    /// `training_x` training observations
    ///
    /// `training_y` training targets
    ///
    /// # Returns
    ///
    /// The calculated weights (coefficients) are returned
    ///
    pub fn fit(
        &mut self,
        training_x: &mut Matrix2d,
        training_y: &mut Matrix2d,
    ) -> Result<Matrix2d, Error> {
        if training_x.get_size().0 != training_y.get_size().0 {
            return Err(Error::InputError("training data size mismatch".to_string()));
        }

        // This is the gradient of least squares loss w.r.t. w
        self.coefficients = training_x
            .transpose()?
            .matmul(training_x)?
            .inverse()?
            .matmul(&training_x.transpose()?)?
            .matmul(&training_y)?;

        return Ok(self.coefficients.clone());
    }

    /// Calculates the prediction given the input
    pub fn predict(&self, x: &mut Matrix2d) -> Result<f64, Error> {
        match x.matmul(&self.coefficients) {
            Ok(mult_matrix) => {
                let mult = mult_matrix.get_elements()[0][0];
                Ok(mult)
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::linalg::Matrix2d;

    use super::LinearRegression;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn fit_test() {
        // f(a, b) = a + 2 * b
        let training_data_vec = vec![
            vec![1., 2.],
            vec![2., 3.],
            vec![3., 4.],
            vec![4., 5.],
            vec![5., 6.],
        ];
        let mut training_x = Matrix2d::from_vec(&training_data_vec).unwrap();
        let mut training_y = Matrix2d::from_vec(&vec![vec![5., 8., 11., 14., 17.]])
            .unwrap()
            .transpose()
            .unwrap();

        let mut lr = LinearRegression::new();

        let coef = lr.fit(&mut training_x, &mut training_y).unwrap().get_elements();
        assert_approx_eq!(coef[0][0], 1.);
        assert_approx_eq!(coef[1][0], 2.);

        assert_approx_eq!(
            lr.predict(&mut Matrix2d::from_vec(&vec![vec![15., 80.]]).unwrap())
                .unwrap(),
            175.
        );
    }
}
