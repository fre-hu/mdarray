use crate::index::axis::{Axis, Get, Keep};
use crate::layout::{Layout, Strided};
use crate::shape::{DynRank, Shape};

/// Array permutation trait, for array types after permutation of dimensions.
pub trait Permutation {
    /// Shape after permuting dimensions.
    type Shape<S: Shape>: Shape;

    /// Layout after permuting dimensions.
    type Layout<L: Layout>: Layout;
}

impl<X: Axis> Permutation for (X,) {
    type Shape<S: Shape> = (Get<X, S>,);
    type Layout<L: Layout> = L;
}

macro_rules! impl_permutation {
    (($($ij:tt),+), $k:tt, ($($xy:tt),+), $z:tt) => {
        impl<$($xy: Axis,)+ $z: Axis> Permutation for ($($xy,)+ $z)
        where
            ($($xy,)+): Permutation
        {
            type Shape<S: Shape> =
                <<($($xy,)+) as Permutation>::Shape<S> as Shape>::Concat<(Get<$z, S>,)>;
            type Layout<L: Layout> =
                Keep<$z, Self::Shape<()>, <($($xy,)+) as Permutation>::Layout<L>>;
        }
    };
}

impl_permutation!((0), 1, (X), Y);
impl_permutation!((0, 1), 2, (X, Y), Z);
impl_permutation!((0, 1, 2), 3, (X, Y, Z), W);
impl_permutation!((0, 1, 2, 3), 4, (X, Y, Z, W), U);
impl_permutation!((0, 1, 2, 3, 4), 5, (X, Y, Z, W, U), V);

impl Permutation for DynRank {
    type Shape<S: Shape> = S::Dyn;
    type Layout<L: Layout> = Strided;
}
