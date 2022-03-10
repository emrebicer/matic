use crate::math::Distance;
use rand::seq::SliceRandom;

// Centroid initialization
pub enum Initialization {
    Random,
    // Kmeans++
    Kmeanspp,
}

pub struct KMeans<D>
where
    D: Distance,
{
    n_clusters: usize,
    initialization: Initialization,
    pub max_iter: usize,
    centroids: Vec<D>,
}

impl<D> KMeans<D>
where
    D: Distance + Copy,
{
    ///
    /// # Parameters
    ///
    /// `n_clusters`: number of clusters, default = 8
    /// `initialization`: method to initalize centroids, default = Initialization::Random
    /// `max_iter`: maximum number of iterations of kmeans algorithm default = 300
    /// `centroids`: use it for manually setting up initial centroids, if None
    ///     or if the amount of given data points is not equal to `n_clusters`
    ///     the centroids will be initialized according to the `initialization`
    /// `dataset`: dataset to be clustered
    ///
    pub fn new(
        n_clusters: Option<usize>,
        initialization: Option<Initialization>,
        max_iter: Option<usize>,
        centroids: Option<Vec<D>>,
    ) -> Self {
        let kmeans = KMeans {
            n_clusters: n_clusters.unwrap_or(8),
            initialization: initialization.unwrap_or(Initialization::Random),
            max_iter: max_iter.unwrap_or(300),
            centroids: centroids.unwrap_or(Vec::new()),
        };

        return kmeans;
    }

    fn initialize_centroids(&mut self, data: &Vec<D>) {
        self.centroids.clear();

        match self.initialization {
            Initialization::Random => {
                // Select random data points from the dataset
                for _ in 0..self.n_clusters {
                    self.centroids.push(
                        data.choose(&mut rand::thread_rng())
                            .expect("Dataset can't be empty")
                            .to_owned(),
                    );
                }
            }
            Initialization::Kmeanspp => {
                // Select the first data point randomly
                self.centroids.push(
                    data.choose(&mut rand::thread_rng())
                        .expect("Dataset can't be empty")
                        .to_owned(),
                );

                // Select the rest of centroids by finding the
                // furthest away data points from the previous centroids
                for _ in 1..self.n_clusters {
                    let mut furthest_data_point = data.first().unwrap();
                    let mut furthest_data_point_distance = f64::MIN;

                    for data_point in data {
                        // Find the minimum distance between any centroid and this data point
                        let minimum_distance =
                            self.centroids.iter().fold(f64::MAX, |min_dist, centroid| {
                                min_dist.min(centroid.distance(*data_point))
                            });

                        if minimum_distance > furthest_data_point_distance {
                            furthest_data_point = data_point;
                            furthest_data_point_distance = minimum_distance;
                        }
                    }

                    self.centroids.push(*furthest_data_point);
                }
            }
        }
    }

    // Computes the cluster centroids
    // Return centroids and a vector that has the assigned centroid indexes
    pub fn fit(&mut self, data: &Vec<D>) -> (Vec<D>, Vec<usize>) {
        if self.centroids.len() != self.n_clusters {
            self.initialize_centroids(data);
        }

        unimplemented!();
    }

    // Assigns the given data to a cluster
    // Returns the cluster index
    pub fn predict(&self, x: D) -> usize {
        // Find the closests centroid to the input
        let mut min_distance = f64::MAX;
        let mut cluster_index = 0;

        for (index, centroid) in self.centroids.iter().enumerate() {
            let current_distance = x.distance(*centroid);
            if current_distance < min_distance {
                min_distance = current_distance;
                cluster_index = index;
            }
        }

        return cluster_index;
    }

    // Assigns the given batch of data to clusters
    // Returns a vector of cluster indexs in the given order
    pub fn predict_batch(&self, x: Vec<D>) -> Vec<usize> {
        let mut predictions = Vec::new();
        for data_point in x {
            predictions.push(self.predict(data_point));
        }
        return predictions;
    }
}

#[cfg(test)]
mod tests {

    use super::{Initialization, KMeans};
    use crate::math::Point2d;

    #[test]
    fn random_centroid_init() {
        let mut dataset = Vec::new();
        for i in 0..50 {
            dataset.push(Point2d {
                x: i as f64,
                y: i as f64,
            })
        }
        let mut kmeans = KMeans::new(Some(5), Some(Initialization::Random), None, None);

        kmeans.fit(&dataset);
        assert_eq!(kmeans.centroids.len(), 5);

        for centroid in kmeans.centroids {
            assert!(dataset.contains(&centroid));
        }
    }

    #[test]
    fn kmeanspp_centroid_init() {
        let mut dataset = Vec::new();
        for i in 0..50 {
            dataset.push(Point2d {
                x: i as f64,
                y: i as f64,
            })
        }
        let mut kmeans = KMeans::new(Some(5), Some(Initialization::Kmeanspp), None, None);

        kmeans.fit(&dataset);
        assert_eq!(kmeans.centroids.len(), 5);

        for centroid in kmeans.centroids {
            assert!(dataset.contains(&centroid));
        }
    }

    #[test]
    fn predict() {
        let mut dataset = Vec::<Point2d>::new();
        for i in 0..50 {
            dataset.push(Point2d {
                x: i as f64,
                y: i as f64,
            })
        }
        let centroids = vec![
            Point2d { x: 5.0, y: 5.0 },
            Point2d { x: 15.0, y: 15.0 },
            Point2d { x: 25.0, y: 25.0 },
            Point2d { x: 35.0, y: 35.0 },
            Point2d { x: 45.0, y: 45.0 },
        ];
        let kmeans = KMeans::new(Some(5), Some(Initialization::Random), None, Some(centroids));

        let batch = vec![
            Point2d { x: 5.0, y: 5.0 },
            Point2d { x: 9.0, y: 9.0 },
            Point2d { x: 19.0, y: 19.0 },
            Point2d { x: 21.0, y: 20.0 },
            Point2d { x: 30.0, y: 30.1 },
            Point2d { x: 100.0, y: 100.0 },
        ];
        let mut result;
        result = kmeans.predict(*batch.get(0).unwrap());
        assert_eq!(result, 0);
        result = kmeans.predict(*batch.get(1).unwrap());
        assert_eq!(result, 0);
        result = kmeans.predict(*batch.get(2).unwrap());
        assert_eq!(result, 1);
        result = kmeans.predict(*batch.get(3).unwrap());
        assert_eq!(result, 2);
        result = kmeans.predict(*batch.get(4).unwrap());
        assert_eq!(result, 3);
        result = kmeans.predict(*batch.get(5).unwrap());
        assert_eq!(result, 4);

        let correct_labels = vec![0, 0, 1, 2, 3, 4];
        assert_eq!(correct_labels, kmeans.predict_batch(batch));
    }
}
