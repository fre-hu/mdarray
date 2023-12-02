use crate::dim::{Const, Dim};
use crate::layout::{Dense, Flat, Layout};

/// Array axis trait, for subarray types when iterating over or splitting along a dimension.
pub trait Axis<D: Dim> {
    /// Layout when removing the other dimensions.
    type Keep<L: Layout>: Layout;

    /// Layout with the dimension removed.
    type Remove<L: Layout>: Layout;

    /// Layout when splitting along the dimension.
    type Split<L: Layout>: Layout;
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*), $keep:ty, $remove:ty, $split:ty) => {
        $(
            impl Axis<Const<$n>> for Const<$k> {
                type Keep<L: Layout> = $keep;
                type Remove<L: Layout> = $remove;
                type Split<L: Layout> = $split;
            }
        )*
    }
}

impl_axis!((1), (0), L, Dense, L);

impl_axis!((2), (0), L::Uniform, Flat, L::NonUniform);
impl_axis!((2), (1), Flat, L::Uniform, L);

impl_axis!((3, 4, 5, 6), (0, 0, 0, 0), L::Uniform, L::NonUnitStrided, L::NonUniform);
impl_axis!((3, 4, 5, 6), (2, 3, 4, 5), Flat, L, L);

impl_axis!((3), (1), Flat, L::NonUniform, L::NonUniform);
impl_axis!((4, 4), (1, 2), Flat, L::NonUniform, L::NonUniform);
impl_axis!((5, 5, 5), (1, 2, 3), Flat, L::NonUniform, L::NonUniform);
impl_axis!((6, 6, 6, 6), (1, 2, 3, 4), Flat, L::NonUniform, L::NonUniform);
