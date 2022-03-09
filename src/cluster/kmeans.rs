use crate::math::Distance;

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
    pub dataset: Vec<D>,
}

impl<D> KMeans<D>
where
    D: Distance,
{
    ///
    /// `n_clusters`: number of clusters, default = 8
    /// `initialization`: method to initalize centroids,
    ///     default = Initialization::Random
    /// `max_iter`: maximum number of iterations of kmeans algorithm
    ///     default = 300
    /// `centroids`: use it for manually setting up initial centroids, if None
    ///     or if the amount of given data points is not equal to `n clusters`
    ///     the centroids will be initialized according to the `initialization`
    /// `dataset`: dataset to be clustered
    ///
    pub fn new(
        n_clusters: Option<usize>,
        initialization: Option<Initialization>,
        max_iter: Option<usize>,
        centroids: Option<Vec<D>>,
        dataset: Vec<D>,
    ) -> Self {
        let mut kmeans = KMeans {
            n_clusters: n_clusters.unwrap_or(8),
            initialization: initialization.unwrap_or(Initialization::Random),
            max_iter: max_iter.unwrap_or(300),
            centroids: centroids.unwrap_or(Vec::new()),
            dataset
        };

        if kmeans.centroids.len() != kmeans.n_clusters {
            kmeans.initialize_centroids();
        }

        return kmeans
    }

    fn initialize_centroids(&mut self) {

        self.centroids.clear();

        match self.initialization {
            Initialization::Random => {
                unimplemented!();
            },
            Initialization::Kmeanspp=> {
                unimplemented!();
            }
        }

    }

    // Computes the cluster centers
    //pub fit()

    // Assigns the data to clusters
    //pub predict()

    // Computes the centers, and assigns the data to clusters
    //pub fit_predict()
}

//pub fn fit()
