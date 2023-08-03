use crate::dim::{Const, Dim};
use crate::layout::Layout;

/// Array axis trait, for subarray types when iterating over or splitting along a dimension.
pub trait Axis<D: Dim> {
    /// Layout with the dimension removed.
    type Remove<L: Layout>: Layout;

    /// Layout when splitting along the dimension.
    type Split<L: Layout>: Layout;
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*), $remove:ty, $split:ty) => {
        $(
            impl Axis<Const<$n>> for Const<$k> {
                type Remove<L: Layout> = <Const<{ $n - 1 }> as Dim>::Layout<$remove>;
                type Split<L: Layout> = $split;
            }
        )*
    }
}

impl_axis!((1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5), L, L);
impl_axis!((2, 3, 4, 5, 6), (0, 0, 0, 0, 0), L::NonUnitStrided, L::NonUniform);
impl_axis!((3, 4, 5, 6), (1, 1, 1, 1), L::NonUniform, L::NonUniform);
impl_axis!((4, 5, 6), (2, 2, 2), L::NonUniform, L::NonUniform);
impl_axis!((5, 6), (3, 3), L::NonUniform, L::NonUniform);
impl_axis!((6), (4), L::NonUniform, L::NonUniform);
