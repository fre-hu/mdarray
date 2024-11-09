use crate::index::axis::{Axis, Keep};
use crate::layout::Layout;
use crate::shape::Shape;

/// Array permutation trait, for array types after permutation of dimensions.
pub trait Permutation {
    /// Shape after permuting dimensions.
    type Shape<S: Shape>: Shape;

    /// Layout after permuting dimensions.
    type Layout<L: Layout>: Layout;

    #[doc(hidden)]
    fn index_mask(rank: usize) -> usize;
}

impl<X: Axis> Permutation for (X,) {
    type Shape<S: Shape> = (X::Dim<S>,);
    type Layout<L: Layout> = L;

    fn index_mask(rank: usize) -> usize {
        1 << X::index(rank)
    }
}

macro_rules! impl_permutation {
    (($($xy:tt),+), $z:tt) => {
        impl<$($xy: Axis,)+ $z: Axis> Permutation for ($($xy,)+ $z)
        where
            ($($xy,)+): Permutation
        {
            type Shape<S: Shape> =
                <<($($xy,)+) as Permutation>::Shape<S> as Shape>::Concat<($z::Dim<S>,)>;
            type Layout<L: Layout> =
                Keep<$z, Self::Shape<()>, <($($xy,)+) as Permutation>::Layout<L>>;

            fn index_mask(rank: usize) -> usize {
                <($($xy,)+) as Permutation>::index_mask(rank) | (1 << $z::index(rank))
            }
        }
    };
}

impl_permutation!((X), Y);
impl_permutation!((X, Y), Z);
impl_permutation!((X, Y, Z), W);
impl_permutation!((X, Y, Z, W), U);
impl_permutation!((X, Y, Z, W, U), V);
