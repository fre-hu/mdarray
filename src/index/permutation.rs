use crate::dim::{Const, Dim};
use crate::index::axis::{Axis, Nth};
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
    <Nth<K> as Axis>::Insert<<Nth<N> as Axis>::Dim<S>, <P as Permutation>::Shape<S>>;

impl Permutation for (Const<0>,) {
    type Shape<S: Shape> = (S::Head,);
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
impl_permutation!(2, 1, (X), (Z), Strided);
impl_permutation!(2, 2, (X, Y), (), <(X, Y) as Permutation>::Layout<L>);

impl_permutation!(3, 0, (), (Y, Z, W), Strided);
impl_permutation!(3, 1, (X), (Z, W), Strided);
impl_permutation!(3, 2, (X, Y), (W), Strided);
impl_permutation!(3, 3, (X, Y, Z), (), <(X, Y, Z) as Permutation>::Layout<L>);

impl_permutation!(4, 0, (), (Y, Z, W, U), Strided);
impl_permutation!(4, 1, (X), (Z, W, U), Strided);
impl_permutation!(4, 2, (X, Y), (W, U), Strided);
impl_permutation!(4, 3, (X, Y, Z), (U), Strided);
impl_permutation!(4, 4, (X, Y, Z, W), (), <(X, Y, Z, W) as Permutation>::Layout<L>);

impl_permutation!(5, 0, (), (Y, Z, W, U, V), Strided);
impl_permutation!(5, 1, (X), (Z, W, U, V), Strided);
impl_permutation!(5, 2, (X, Y), (W, U, V), Strided);
impl_permutation!(5, 3, (X, Y, Z), (U, V), Strided);
impl_permutation!(5, 4, (X, Y, Z, W), (V), Strided);
impl_permutation!(5, 5, (X, Y, Z, W, U), (), <(X, Y, Z, W, U) as Permutation>::Layout<L>);
