use crate::dim::{Dim, Dyn};
use crate::layout::Layout;
use crate::mapping::{DenseMapping, Mapping};
use crate::shape::Shape;

/// Array axis trait, for subarray shapes.
pub trait Axis {
    /// Corresponding dimension.
    type Dim<S: Shape>: Dim;

    /// Shape for the previous dimensions excluding the current dimension.
    type Init<S: Shape>: Shape;

    /// Shape for the next dimensions excluding the current dimension.
    type Rest<S: Shape>: Shape;

    #[doc(hidden)]
    fn index(rank: usize) -> usize;

    #[doc(hidden)]
    fn keep<M: Mapping>(
        mapping: &M,
    ) -> <Keep<Self, M::Shape, M::Layout> as Layout>::Mapping<(Self::Dim<M::Shape>,)> {
        let index = Self::index(mapping.rank());

        Mapping::prepend_dim(&DenseMapping::new(()), mapping.dim(index), mapping.stride(index))
    }

    #[doc(hidden)]
    fn remove<M: Mapping>(
        mapping: &M,
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Remove<Self, M::Shape>> {
        Mapping::remove_dim::<M>(mapping, Self::index(mapping.rank()))
    }

    #[doc(hidden)]
    fn resize<M: Mapping>(
        mapping: &M,
        new_size: usize,
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Resize<Self, M::Shape>> {
        Mapping::resize_dim::<M>(mapping, Self::index(mapping.rank()), new_size)
    }
}

/// Axis type for the N:th dimension.
pub struct Nth<const N: usize>;

/// Shape when removing the dimension for the specified axis.
pub type Remove<A, S> = <<A as Axis>::Init<S> as Shape>::Concat<<A as Axis>::Rest<S>>;

/// Shape when resizing the dimension for the specified axis.
pub type Resize<A, S> =
    <<A as Axis>::Init<S> as Shape>::Concat<<<A as Axis>::Rest<S> as Shape>::Prepend<Dyn>>;

/// Layout when keeping the dimension for the specified axis.
pub type Keep<A, S, L> = <<A as Axis>::Rest<S> as Shape>::Layout<L>;

/// Layout when removing or resizing the dimension for the specified axis.
pub type Split<A, S, L> = <<A as Axis>::Init<S> as Shape>::Layout<L>;

//
// The tables below give the resulting layout depending on the rank and axis.
//
// Keep<A, S, L>:
//
// Rank \ Axis  0           1           2           3
// -------------------------------------------------------------
// 1            L           -           -           -
// 2            Strided     L           -           -
// 3            Strided     Strided     L           -
// 4            Strided     Strided     Strided     L
// ...
// DynRank      Strided     Strided     Strided     Strided
//
// Split<A, S, L>:
//
// Rank \ Axis  0           1           2           3
// -------------------------------------------------------------
// 1            L           -           -           -
// 2            L           Strided     -           -
// 3            L           Strided     Strided     -
// 4            L           Strided     Strided     Strided
// ...
// DynRank      L           Strided     Strided     Strided
//

impl Axis for Nth<0> {
    type Dim<S: Shape> = S::Head;

    type Init<S: Shape> = ();
    type Rest<S: Shape> = S::Tail;

    fn index(rank: usize) -> usize {
        assert!(rank > 0, "invalid dimension");

        0
    }
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*)) => {
        $(
            impl Axis for Nth<$n> {
                type Dim<S: Shape> = <Nth<$k> as Axis>::Dim<S::Tail>;

                type Init<S: Shape> = <<Nth<$k> as Axis>::Init<S::Tail> as Shape>::Prepend<S::Head>;
                type Rest<S: Shape> = <Nth<$k> as Axis>::Rest<S::Tail>;

                fn index(rank: usize) -> usize {
                    assert!(rank > $n, "invalid dimension");

                    $n
                }
            }
        )*
    };
}

impl_axis!((1, 2, 3, 4, 5), (0, 1, 2, 3, 4));
