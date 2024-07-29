use crate::dim::{Const, Dim};
use crate::index::axis::{Axis, Inner};
use crate::layout::{Layout, Strided};
use crate::shape::Shape;

/// Array permutation trait, for array types after permutation of dimensions.
pub trait Permutation {
    /// Shape after permuting dimensions.
    type Shape<S: Shape>: Shape;

    /// Layout after permuting dimensions.
    type Layout<L: Layout>: Layout;
}

type Insert<P, S, const N: usize, const K: usize> =
    <Inner<K> as Axis>::Insert<<Inner<N> as Axis>::Dim<S>, <P as Permutation>::Shape<S>>;

impl Permutation for (Const<0>,) {
    type Shape<S: Shape> = S::Head;
    type Layout<L: Layout> = L;
}

macro_rules! impl_permutation {
    ($n:tt, $k:tt, ($($x:tt),*), ($($z:tt),*), $layout:ty) => {
        impl<$($x: Dim,)* $($z: Dim),*> Permutation for ($($x,)* Const<$n>, $($z),*)
        where
            ($($x,)* $($z,)*): Permutation
        {
            type Shape<S: Shape> = Insert<($($x,)* $($z,)*), S, $n, $k>;
            type Layout<L: Layout> = $layout;
        }
    };
}

impl_permutation!(1, 0, (), (Y), Strided);
impl_permutation!(1, 1, (X), (), L);

impl_permutation!(2, 0, (), (Y, Z), Strided);
impl_permutation!(2, 1, (X), (Z), <(X, Z) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(2, 2, (X, Y), (), <(X, Y) as Permutation>::Layout<L>);

impl_permutation!(3, 0, (), (Y, Z, W), Strided);
impl_permutation!(3, 1, (X), (Z, W), <(X, Z, W) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(3, 2, (X, Y), (W), <(X, Y, W) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(3, 3, (X, Y, Z), (), <(X, Y, Z) as Permutation>::Layout<L>);

impl_permutation!(4, 0, (), (Y, Z, W, U), Strided);
impl_permutation!(4, 1, (X), (Z, W, U), <(X, Z, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, 2, (X, Y), (W, U), <(X, Y, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, 3, (X, Y, Z), (U), <(X, Y, Z, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, 4, (X, Y, Z, W), (), <(X, Y, Z, W) as Permutation>::Layout<L>);

impl_permutation!(5, 0, (), (Y, Z, W, U, V), Strided);
impl_permutation!(5, 1, (X), (Z, W, U, V), <(X, Z, W, U, V) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, 2, (X, Y), (W, U, V), <(X, Y, W, U, V) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, 3, (X, Y, Z), (U, V), <(X, Y, Z, U, V) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, 4, (X, Y, Z, W), (V), <(X, Y, Z, W, V) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, 5, (X, Y, Z, W, U), (), <(X, Y, Z, W, U) as Permutation>::Layout<L>);
