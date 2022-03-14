use crate::error::Error;
use std::iter::zip;

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

    use super::dot;
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
}
