use crate::index::axis::{Axis, Get};
use crate::layout::{Layout, Strided};
use crate::shape::{DynRank, Shape};

/// Array permutation trait, for array types after permutation of dimensions.
pub trait Permutation {
    /// Shape after permuting dimensions.
    type Shape<S: Shape>: Shape;

    /// Layout after permuting dimensions.
    type Layout<L: Layout>: Layout;

    #[doc(hidden)]
    type Init: Shape;
}

impl<X: Axis> Permutation for (X,) {
    type Shape<S: Shape> = (Get<X, S>,);
    type Layout<L: Layout> = L;

    type Init = X::Init<()>;
}

macro_rules! impl_permutation {
    (($($jk:tt),+), ($($yz:tt),+)) => {
        impl<X: Axis $(,$yz: Axis)+> Permutation for (X $(,$yz)+)
        where
            ($($yz,)+): Permutation
        {
            type Shape<S: Shape> =
                <<($($yz,)+) as Permutation>::Shape<S> as Shape>::Prepend<Get<X, S>>;
            type Layout<L: Layout> = <Self::Init as Shape>::Layout<L>;

            type Init =
                <<<($($yz,)+) as Permutation>::Init as Shape>::Tail as Shape>::Merge<X::Init<()>>;
            }
    };
}

impl_permutation!((1), (Y));
impl_permutation!((1, 2), (Y, Z));
impl_permutation!((1, 2, 3), (Y, Z, W));
impl_permutation!((1, 2, 3, 4), (Y, Z, W, U));
impl_permutation!((1, 2, 3, 4, 5), (Y, Z, W, U, V));

impl Permutation for DynRank {
    type Shape<S: Shape> = S::Dyn;
    type Layout<L: Layout> = Strided;

    type Init = Self;
}
