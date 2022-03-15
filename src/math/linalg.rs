use crate::error::Error;
use std::iter::zip;

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix2d {
    size: (usize, usize),
    elements: Vec<Vec<f64>>,
}

impl Matrix2d {
    /// Creates a zero matrix with given size
    ///
    /// # Parameters
    /// `size` both values in this tuple should be positive
    pub fn new(size: (usize, usize)) -> Result<Self, Error> {
        if size.0 == 0 || size.1 == 0 {
            return Err(Error::InputError("size must be positive"));
        }
        let mut elements = vec![];
        for _ in 0..size.0 {
            let mut row = vec![];
            for _ in 0..size.1 {
                row.push(0.);
            }
            elements.push(row);
        }

        Ok(Matrix2d { size, elements })
    }

    /// Creates a matrix from the given elements
    ///
    /// # Parameters
    /// `elements` should not be empty
    pub fn from_vec(elements: &Vec<Vec<f64>>) -> Result<Self, Error> {
        if elements.is_empty() {
            return Err(Error::InputError("size must be positive"));
        }

        // Make sure the training data contains the same sized vectors
        let row_size = elements.len();
        let column_size = elements.first().unwrap().len();

        for column in elements.iter() {
            if column.len() != column_size {
                return Err(Error::InputError(
                    "elements must include same sized vectors",
                ));
            }
        }

        Ok(Matrix2d {
            size: (row_size, column_size),
            elements: elements.clone(),
        })
    }

    pub fn get_size(&self) -> (usize, usize) {
        return self.size;
    }

    pub fn get_elements(&self) -> Vec<Vec<f64>>{
        return self.elements.clone();
    }

    pub fn is_square_matrix(&self) -> bool {
        return self.size.0 == self.size.1;
    }

    pub fn inverse(&self) -> Result<Self, Error> {
        let inversed = inverse_2d(&self.elements)?;
        return Self::from_vec(&inversed);
    }

    pub fn transpose(&self) -> Result<Self, Error> {
        let transposed = transpose_2d(&self.elements)?;
        return Self::from_vec(&transposed);
    }

    /// Calculates the output of matrix multiplication from `first` * `second`
    /// The number of columns in the `first` matrix must match the number of
    /// rows in the `second` matrix
    pub fn matmul(&self, second: &Self) -> Result<Self, Error> {
        if self.size.1 != second.size.0 {
            return Err(Error::InputError(""));
        }

        let mut out = Matrix2d::new((self.size.0, second.size.1))?;

        for row in 0..out.size.0 {
            for col in 0..out.size.1 {
                let mut val = 0.;
                for i in 0..second.size.0 {
                    val += self.elements[row][i] * second.elements[i][col];
                }
                out.elements[row][col] = val;
            }
        }

        return Ok(out);
    }
}

pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, Error> {
    if a.len() != b.len() {
        return Err(Error::InputError("a and b must have the same size"));
    }

    return Ok(zip(a, b)
        .map(|val| val.0 * val.1)
        .fold(0., |acc, x| acc + x));
}

/// Calculates the inverse of the given matrix
/// NOTE: it uses Gauss Jordan method to calculate the inverse, which can't
/// calculate the inverse of a matrix if there is a 0 on the diagonal. As a
/// temporary workaround the 0 values on the diagonal are replaced with a
/// very small float value.
/// TODO: implement a proper inverse function that work on every possible input
pub fn inverse_2d(matrix: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, Error> {
    // Input matrix must be a square matrix
    if matrix.is_empty() {
        return Err(Error::InputError("matrix can't be empty"));
    }

    let mut inverse = vec![];
    let n = matrix.len();
    for (i, vals) in matrix.iter().enumerate() {
        if vals.len() != n {
            return Err(Error::InputError("input must be a square matrix"));
        }
        let mut row = vec![];
        for val in vals {
            row.push(*val);
        }
        for j in 0..n {
            if i == j {
                row.push(1.);
            } else {
                row.push(0.);
            }
        }
        inverse.push(row);
    }

    for i in 0..n {
        if inverse[i][i] == 0. {
            //return Err(Error::InputError("Diogonal elements should not contain 0"));
            inverse[i][i] = 0.00000001;
        }
        for j in 0..n {
            if i != j {
                let ratio = inverse[j][i] / inverse[i][i];
                for k in 0..2 * n {
                    inverse[j][k] -= ratio * inverse[i][k];
                }
            }
        }
    }

    for i in 0..n {
        for j in n..2 * n {
            inverse[i][j] /= inverse[i][i];
        }
    }

    let mut result = vec![];
    for i in 0..n {
        let mut inner = vec![];
        for j in n..2 * n {
            inner.push(inverse[i][j]);
        }
        result.push(inner);
    }
    return Ok(result);
}
pub fn transpose_2d(matrix: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, Error> {
    if matrix.is_empty() {
        return Err(Error::InputError("matrix can't be empty"));
    }

    let m = matrix.len();
    let n = matrix.first().unwrap().len();

    for vals in matrix.iter() {
        if vals.len() != n {
            return Err(Error::InputError(
                "all rows in the matrix should have the same length",
            ));
        }
    }

    let mut transpose = vec![];
    for i in 0..n {
        let mut inner = vec![];
        for j in 0..m {
            inner.push(matrix[j][i]);
        }
        transpose.push(inner);
    }

    return Ok(transpose);
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::{dot, Matrix2d};
    use crate::{
        error::Error,
        math::linalg::{inverse_2d, transpose_2d},
    };

    #[test]
    fn dot_test() {
        assert_eq!(
            dot(vec![1., 3., -5.].as_ref(), vec![4., -2., -1.].as_ref()),
            Ok(3.)
        );
        assert_eq!(
            dot(vec![1., 3.].as_ref(), vec![4., -2., -1.].as_ref()),
            Err(Error::InputError("a and b must have the same size"))
        );
        assert_eq!(
            dot(vec![1., 3., -5.].as_ref(), vec![4., -2.].as_ref()),
            Err(Error::InputError("a and b must have the same size"))
        );
        assert_eq!(dot(vec![].as_ref(), vec![].as_ref()), Ok(0.));
    }

    #[test]
    fn transpose_2d_test() {
        let matrix = vec![vec![-2., 5., 6.], vec![5., 2., 7.]];
        let transposed_correct = vec![vec![-2., 5.], vec![5., 2.], vec![6., 7.]];
        assert_eq!(transpose_2d(&matrix), Ok(transposed_correct));

        let matrix = vec![vec![-2., 0.], vec![0., 5.]];
        let transposed_correct = vec![vec![-2., 0.], vec![0., 5.]];
        assert_eq!(transpose_2d(&matrix), Ok(transposed_correct));

        let matrix = vec![vec![-2., 5., 6., -230.], vec![5., 2., 7., -24.]];
        assert_eq!(transpose_2d(&transpose_2d(&matrix).unwrap()), Ok(matrix));
    }

    #[test]
    fn inverse_2d_test() {
        let matrix = vec![vec![1., -2.], vec![3., 4.]];
        let inversed_correct = vec![vec![0.4, 0.2], vec![-0.3, 0.1]];
        compare_float_vec(&inverse_2d(&matrix).unwrap(), &inversed_correct);

        let matrix = vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 5., 9.]];
        let inversed_correct = vec![
            vec![-5. / 6., 1. / 6., 1. / 6.],
            vec![-1. / 3., 2. / 3., -1. / 3.],
            vec![5. / 6., -1. / 2., 1. / 6.],
        ];
        compare_float_vec(&inverse_2d(&matrix).unwrap(), &inversed_correct);

        let matrix = vec![vec![0., 2., 3.], vec![4., 5., 6.], vec![7., 5., 9.]];
        let inversed_correct = vec![
            vec![-5. / 11., 1. / 11., 1. / 11.],
            vec![-2. / 11., 7. / 11., -4. / 11.],
            vec![5. / 11., -14. / 33., 8. / 33.],
        ];
        compare_float_vec(&inverse_2d(&matrix).unwrap(), &inversed_correct);

        let matrix = vec![vec![0., 2., 3.], vec![4., 5., 6.], vec![7., 5., 0.]];
        let inversed_correct = vec![
            vec![-10. / 13., 5. / 13., -1. / 13.],
            vec![14. / 13., -7. / 13., 4. / 13.],
            vec![-5. / 13., 14. / 39., -8. / 39.],
        ];
        compare_float_vec(&inverse_2d(&matrix).unwrap(), &inversed_correct);

        let matrix = vec![vec![0., 2., 3.], vec![4., 0., 6.], vec![7., 5., 0.]];
        let inversed_correct = vec![
            vec![-5. / 24., 5. / 48., 1. / 12.],
            vec![7. / 24., -7. / 48., 1. / 12.],
            vec![5. / 36., 7. / 72., -1. / 18.],
        ];
        compare_float_vec(&inverse_2d(&matrix).unwrap(), &inversed_correct);
    }

    fn compare_float_vec(first: &Vec<Vec<f64>>, second: &Vec<Vec<f64>>) {
        first.iter().zip(second.iter()).for_each(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .for_each(|(&v1, &v2)| assert_approx_eq!(v1, v2))
        });
    }

    #[test]
    fn matrix_size_test() {
        let elems = vec![vec![0., 2., 3.], vec![4., 0., 6.]];
        let matrix = Matrix2d::from_vec(&elems).unwrap();
        assert_eq!((2, 3), matrix.size);
        let _ = matrix.inverse();
        let rc = matrix.elements.len();
        let cc = matrix.elements.first().unwrap().len();
        assert_eq!((rc, cc), matrix.size);
        let _ = matrix.transpose();
        let rc = matrix.elements.len();
        let cc = matrix.elements.first().unwrap().len();
        assert_eq!((rc, cc), matrix.size);
    }

    #[test]
    fn matrix_multiplication_test() {
        let first = Matrix2d::from_vec(&vec![vec![1., 5.], vec![2., 3.], vec![1., 7.]]).unwrap();
        let second = Matrix2d::from_vec(&vec![vec![1., 2., 3., 7.], vec![5., 2., 8., 1.]]).unwrap();

        let expected = Matrix2d::from_vec(&vec![
            vec![26., 12., 43., 12.],
            vec![17., 10., 30., 17.],
            vec![36., 16., 59., 14.],
        ]).unwrap();

        assert_eq!(
            Matrix2d::matmul(&first, &second).unwrap(),
            expected
        );

        assert_eq!(
            first.matmul(&second).unwrap(),
            expected
        );
    }
}
