pub mod linalg;
use std::ops::{Add, Div};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

// Euclidean distance for 2 dimensions
impl Distance for Point2d {
    fn distance(&self, other: Point2d) -> f64 {
        ((&self.x - other.x).powf(2.0) + (&self.y - other.y).powf(2.0)).sqrt()
    }
}

impl Average for Point2d {
    fn average(data: &Vec<Self>) -> Self {
        if data.len() == 0 {
            return Point2d {
                x: 0.0,
                y: 0.0,
            };
        }

        let mut x_total = 0.0;
        let mut y_total = 0.0;

        data.iter().for_each(|p| {
            x_total += p.x;
            y_total += p.y;
        });

        Point2d {
            x: x_total / data.len() as f64,
            y: y_total / data.len() as f64,
        }
    }
}

impl Add for Point2d {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Div<usize> for Point2d {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        if rhs == 0 {
            panic!("Cannot divide by zero");
        }
        Self {
            x: self.x / rhs as f64,
            y: self.y / rhs as f64,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Add for Point3d {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Div<usize> for Point3d {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        if rhs == 0 {
            panic!("Cannot divide by zero");
        }
        Self {
            x: self.x / rhs as f64,
            y: self.y / rhs as f64,
            z: self.z / rhs as f64,
        }
    }
}

// Euclidean distance for 3 dimensions
impl Distance for Point3d {
    fn distance(&self, other: Point3d) -> f64 {
        ((&self.x - other.x).powf(2.0)
            + (&self.y - other.y).powf(2.0)
            + (&self.z - other.z).powf(2.0))
        .sqrt()
    }
}

pub trait Distance {
    fn distance(&self, other: Self) -> f64;
}

pub trait Average {
    fn average(data: &Vec<Self>) -> Self
    where
        Self: Sized;
}
