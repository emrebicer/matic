use crate::math::Distance;

#[derive(PartialEq, Debug)]
struct Neighbor<L> {
    distance: f64,
    label: L,
}

/// Predict the label of `x` given `dataset` and `k`
pub fn predict<F, L>(dataset: &Vec<(F, L)>, x: F, k: usize) -> Option<L>
where
    F: Distance + Copy,
    L: Eq + Copy + std::fmt::Debug,
{
    if dataset.len() < k {
        return None;
    }

    if k < 1 {
        return None;
    }

    // Find the closest neighbors to x
    let mut closest_neighbors = Vec::new();
    dataset.into_iter().for_each(|data_point| {
        let current_distance = data_point.0.distance(x);

        if closest_neighbors.len() < k {
            closest_neighbors.push(Neighbor {
                distance: current_distance,
                label: data_point.1,
            });
        } else {
            // Look for the longest distance in the current neighbors
            let longest_distance = closest_neighbors
                .iter()
                .fold(f64::MIN, |max_dist, ins| max_dist.max(ins.distance));

            if current_distance < longest_distance {
                // Update the neighbor
                let index = closest_neighbors
                    .iter()
                    .position(|neighbor| neighbor.distance == longest_distance)
                    .expect("this element must exist in the vector");

                closest_neighbors.remove(index);
                closest_neighbors.push(Neighbor {
                    distance: current_distance,
                    label: data_point.1,
                });
            }
        }
    });

    // Find the majority of labels
    let mut majority_count = 0;
    let mut majority_label = closest_neighbors
        .first()
        .expect("the must be at least 1 closest neighbor")
        .label;

    for neighbor_f in &closest_neighbors {
        let mut current_label_count = 1;
        let current_label = neighbor_f.label;
        for neighbor_s in &closest_neighbors {
            if neighbor_f != neighbor_s && current_label == neighbor_s.label {
                current_label_count += 1;
                if current_label_count > majority_count {
                    majority_count = current_label_count;
                    majority_label = current_label;
                }
            }
        }
    }

    return Some(majority_label);
}

#[cfg(test)]
mod tests {

    use crate::math::{Point2d, Point3d};
    use super::predict;

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    enum MyLabels {
        FirstLabel,
        SecondLabel,
    }

    #[test]
    fn small_dataset_2d() {
        let mut dataset_2d = Vec::new();
        for i in 0..20 {
            let label = if i < 10 {
                MyLabels::FirstLabel
            } else {
                MyLabels::SecondLabel
            };

            dataset_2d.push((
                Point2d {
                    x: i as f64,
                    y: i as f64,
                },
                label,
            ));
        }

        let x1 = Point2d { x: 2.0, y: 2.0 };
        let x2 = Point2d { x: 12.0, y: 12.0 };
        let x3 = Point2d { x: 10.1, y: 10.1 };

        let output_1 = predict(&dataset_2d, x1, 5);
        let output_2 = predict(&dataset_2d, x2, 3);
        let output_3 = predict(&dataset_2d, x2, 21);
        let output_4 = predict(&dataset_2d, x2, 0);
        let output_5 = predict(&dataset_2d, x3, 1);
        let output_6 = predict(&dataset_2d, x3, 9);

        assert_eq!(output_1, Some(MyLabels::FirstLabel));
        assert_eq!(output_2, Some(MyLabels::SecondLabel));
        assert_eq!(output_3, None);
        assert_eq!(output_4, None);
        assert_eq!(output_2, Some(MyLabels::SecondLabel));
        assert_eq!(output_5, Some(MyLabels::SecondLabel));
        assert_eq!(output_6, Some(MyLabels::SecondLabel));
    }

    #[test]
    fn small_dataset_3d() {
        let mut dataset_3d = Vec::new();
        for i in 0..20 {
            let label = if i < 10 {
                MyLabels::FirstLabel
            } else {
                MyLabels::SecondLabel
            };

            dataset_3d.push((
                Point3d {
                    x: i as f64,
                    y: i as f64,
                    z: i as f64,
                },
                label,
            ));
        }

        let x1 = Point3d {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let x2 = Point3d {
            x: 12.0,
            y: 12.0,
            z: 12.0,
        };
        let x3 = Point3d {
            x: 10.1,
            y: 10.1,
            z: 10.1,
        };

        let output_1 = predict(&dataset_3d, x1, 5);
        let output_2 = predict(&dataset_3d, x2, 3);
        let output_3 = predict(&dataset_3d, x2, 21);
        let output_4 = predict(&dataset_3d, x2, 0);
        let output_5 = predict(&dataset_3d, x3, 1);
        let output_6 = predict(&dataset_3d, x3, 9);

        assert_eq!(output_1, Some(MyLabels::FirstLabel));
        assert_eq!(output_2, Some(MyLabels::SecondLabel));
        assert_eq!(output_3, None);
        assert_eq!(output_4, None);
        assert_eq!(output_2, Some(MyLabels::SecondLabel));
        assert_eq!(output_5, Some(MyLabels::SecondLabel));
        assert_eq!(output_6, Some(MyLabels::SecondLabel));
    }
}
