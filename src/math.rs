pub struct Point2d {
    x: f64,
    y: f64,
}

// Euclidean distance for 2 dimensions
impl Distance for Point2d {
    fn distance(&self, other: Point2d) -> f64 {
        ((&self.x - other.x).powf(2.0) + (&self.y - other.y).powf(2.0)).sqrt()
    }
}

pub struct Point3d {
    x: f64,
    y: f64,
    z: f64,
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
