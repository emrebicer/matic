use crate::common::Label;
use crate::math::Distance;

struct Instance<L: Label> {
    distance: f64,
    label: L,
}

impl<L> Instance<L>
where
    L: Label,
{
    fn new(label_type: L) -> Self {
        Instance {
            distance: f64::MAX,
            label: label_type,
        }
    }
}

pub fn predict<L, T, D>(dataset: &Vec<T>, x: T, k: usize) -> Option<L>
where
    T: Label + Distance + Copy,
    L: Label,
{

    dataset.into_iter().for_each(|data_point| {
        let n = Instance::<L> {
            distance: T::distance(*data_point, x),
            label: data_point.label()
        };
    });

    unimplemented!();

    //let label = dataset.into_iter().last().unwrap().label();
    //Some(label)
}
