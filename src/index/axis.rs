use crate::dim::{Const, Dim};
use crate::format::Format;

/// Array axis trait, for subarray types when iterating over or splitting along a dimension.
pub trait Axis<D: Dim> {
    /// Format with the dimension removed.
    type Remove<F: Format>: Format;

    /// Format when splitting along the dimension.
    type Split<F: Format>: Format;
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*), $remove:ty, $split:ty) => {
        $(
            impl Axis<Const<$n>> for Const<$k> {
                type Remove<F: Format> = <Const<{ $n - 1 }> as Dim>::Format<$remove>;
                type Split<F: Format> = $split;
            }
        )*
    }
}

impl_axis!((1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5), F, F);
impl_axis!((2, 3, 4, 5, 6), (0, 0, 0, 0, 0), F::NonUnitStrided, F::NonUniform);
impl_axis!((3, 4, 5, 6), (1, 1, 1, 1), F::NonUniform, F::NonUniform);
impl_axis!((4, 5, 6), (2, 2, 2), F::NonUniform, F::NonUniform);
impl_axis!((5, 6), (3, 3), F::NonUniform, F::NonUniform);
impl_axis!((6), (4), F::NonUniform, F::NonUniform);
