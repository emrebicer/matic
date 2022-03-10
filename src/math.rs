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

#[derive(Copy, Clone)]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
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
