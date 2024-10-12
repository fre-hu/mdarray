use crate::dim::{Dim, Dyn};
use crate::layout::{Layout, Strided};
use crate::mapping::{DenseMapping, Mapping};
use crate::shape::Shape;

/// Array axis trait, for subarray types when removing or resizing dimensions.
pub trait Axis {
    /// Corresponding dimension.
    type Dim<S: Shape>: Dim;

    /// Shape for the other dimensions.
    type Other<S: Shape>: Shape;

    /// Insert the dimension into the shape.
    type Insert<D: Dim, S: Shape>: Shape;

    /// Replace the dimension in the shape.
    type Replace<D: Dim, S: Shape>: Shape;

    /// Layout when removing the other dimensions.
    type Keep<S: Shape, L: Layout>: Layout;

    /// Layout when removing or resizing the dimension.
    type Split<S: Shape, L: Layout>: Layout;

    #[doc(hidden)]
    fn index(rank: usize) -> usize;

    #[doc(hidden)]
    fn keep<M: Mapping>(
        mapping: &M,
    ) -> <Self::Keep<M::Shape, M::Layout> as Layout>::Mapping<(Self::Dim<M::Shape>,)> {
        let index = Self::index(mapping.rank());

        Mapping::prepend_dim(&DenseMapping::new(()), mapping.dim(index), mapping.stride(index))
    }

    #[doc(hidden)]
    fn remove<M: Mapping>(
        mapping: &M,
    ) -> <Self::Split<M::Shape, M::Layout> as Layout>::Mapping<Self::Other<M::Shape>> {
        Mapping::remove_dim::<M>(mapping, Self::index(mapping.rank()))
    }

    #[doc(hidden)]
    fn resize<M: Mapping>(
        mapping: &M,
        new_size: usize,
    ) -> <Self::Split<M::Shape, M::Layout> as Layout>::Mapping<Self::Replace<Dyn, M::Shape>> {
        Mapping::resize_dim::<M>(mapping, Self::index(mapping.rank()), new_size)
    }
}

/// Axis type for the N:th dimension.
pub struct Nth<const N: usize>;

//
// The tables below give the resulting layout depending on the rank and index.
//
// Keep<S, L>:
//
// Rank \ Index 0           1           2           3
// -------------------------------------------------------------
// 1            L           -           -           -
// 2            Strided     L           -           -
// 3            Strided     Strided     L           -
// 4            Strided     Strided     Strided     L
// ...
// DynRank      Strided     Strided     Strided     Strided
//
// Split<S, L>:
//
// Rank \ Index 0           1           2           3
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
    type Other<S: Shape> = S::Tail;

    type Insert<D: Dim, S: Shape> = S::Prepend<D>;
    type Replace<D: Dim, S: Shape> = <S::Tail as Shape>::Prepend<D>;

    type Keep<S: Shape, L: Layout> = S::Layout<L>;
    type Split<S: Shape, L: Layout> = L;

    fn index(rank: usize) -> usize {
        assert!(rank > 0, "invalid dimension");

        0
    }
}

impl Axis for Nth<1> {
    type Dim<S: Shape> = <S::Tail as Shape>::Head;
    type Other<S: Shape> = <<S::Tail as Shape>::Tail as Shape>::Prepend<S::Head>;

    type Insert<D: Dim, S: Shape> = <<S::Tail as Shape>::Prepend<D> as Shape>::Prepend<S::Head>;
    type Replace<D: Dim, S: Shape> =
        <<<S::Tail as Shape>::Tail as Shape>::Prepend<D> as Shape>::Prepend<S::Head>;

    type Keep<S: Shape, L: Layout> = <S::Tail as Shape>::Layout<L>;
    type Split<S: Shape, L: Layout> = Strided;

    fn index(rank: usize) -> usize {
        assert!(rank > 1, "invalid dimension");

        1
    }
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*)) => {
        $(
            impl Axis for Nth<$n> {
                type Dim<S: Shape> = <Nth<$k> as Axis>::Dim<S::Tail>;
                type Other<S: Shape> =
                    <<Nth<$k> as Axis>::Other<S::Tail> as Shape>::Prepend<S::Head>;

                type Insert<D: Dim, S: Shape> =
                    <<Nth<$k> as Axis>::Insert<D, S::Tail> as Shape>::Prepend<S::Head>;
                type Replace<D: Dim, S: Shape> =
                    <<Nth<$k> as Axis>::Replace<D, S::Tail> as Shape>::Prepend<S::Head>;

                type Keep<S: Shape, L: Layout> = <Nth<$k> as Axis>::Keep<S::Tail, L>;
                type Split<S: Shape, L: Layout> = Strided;

                fn index(rank: usize) -> usize {
                    assert!(rank > $n, "invalid dimension");

                    $n
                }
            }
        )*
    };
}

impl_axis!((2, 3, 4, 5), (1, 2, 3, 4));
