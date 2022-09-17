use crate::dim::{Dim, Rank};
use crate::format::Format;
use crate::order::{ColumnMajor, RowMajor};

/// Array axis trait, for subarray types when iterating over or splitting along a dimension.
pub trait Axis<D: Dim> {
    /// Format with the dimension removed.
    type Remove<F: Format>: Format;

    /// Format when splitting along the dimension.
    type Split<F: Format>: Format;
}

/// Type-level constant.
pub struct Const<const N: usize>;

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*), $order:ty, $remove:ty, $split:ty) => {
        $(
            impl Axis<Rank<$n, $order>> for Const<$k> {
                type Remove<F: Format> = <Rank<{ $n - 1 }, $order> as Dim>::Format<$remove>;
                type Split<F: Format> = $split;
            }
        )*
    }
}

impl_axis!((1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5), ColumnMajor, F, F);
impl_axis!((2, 3, 4, 5, 6), (0, 0, 0, 0, 0), ColumnMajor, F::NonUnitStrided, F::NonUniform);
impl_axis!((3, 4, 5, 6), (1, 1, 1, 1), ColumnMajor, F::NonUniform, F::NonUniform);
impl_axis!((4, 5, 6), (2, 2, 2), ColumnMajor, F::NonUniform, F::NonUniform);
impl_axis!((5, 6), (3, 3), ColumnMajor, F::NonUniform, F::NonUniform);
impl_axis!((6), (4), ColumnMajor, F::NonUniform, F::NonUniform);

impl_axis!((1, 2, 3, 4, 5, 6), (0, 0, 0, 0, 0, 0), RowMajor, F, F);
impl_axis!((2, 3, 4, 5, 6), (1, 2, 3, 4, 5), RowMajor, F::NonUnitStrided, F::NonUniform);
impl_axis!((3, 4, 5, 6), (1, 1, 1, 1), RowMajor, F::NonUniform, F::NonUniform);
impl_axis!((4, 5, 6), (2, 2, 2), RowMajor, F::NonUniform, F::NonUniform);
impl_axis!((5, 6), (3, 3), RowMajor, F::NonUniform, F::NonUniform);
impl_axis!((6), (4), RowMajor, F::NonUniform, F::NonUniform);
