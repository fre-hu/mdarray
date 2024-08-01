use crate::dim::{Dim, Dyn};
use crate::layout::{Flat, Layout};
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

    /// Layout with the dimension removed.
    type Remove<S: Shape, L: Layout>: Layout;

    /// Layout when resizing the dimension.
    type Resize<S: Shape, L: Layout>: Layout;

    #[doc(hidden)]
    fn index(rank: usize) -> usize;

    #[doc(hidden)]
    fn keep<M: Mapping>(
        mapping: M,
    ) -> <Self::Keep<M::Shape, M::Layout> as Layout>::Mapping<Self::Dim<M::Shape>> {
        let index = Self::index(M::Shape::RANK);

        Mapping::add_dim(DenseMapping::new(()), mapping.dim(index), mapping.stride(index))
    }

    #[doc(hidden)]
    fn remove<M: Mapping>(
        mapping: M,
    ) -> <Self::Remove<M::Shape, M::Layout> as Layout>::Mapping<Self::Other<M::Shape>> {
        Mapping::remove_dim::<M>(mapping, Self::index(M::Shape::RANK))
    }

    #[doc(hidden)]
    fn resize<M: Mapping>(
        mapping: M,
        new_size: usize,
    ) -> <Self::Resize<M::Shape, M::Layout> as Layout>::Mapping<Self::Replace<Dyn, M::Shape>> {
        Mapping::resize_dim::<M>(mapping, Self::index(M::Shape::RANK), new_size)
    }
}

/// Inner axis type, counter from the innermost dimension.
pub struct Inner<const N: usize>;

/// Outer axis type, for the outermost dimension.
pub struct Outer;

//
// The tables below give the resulting layout depending on the rank and index.
//
// Keep<L>:
//
// Rank \ Index 0               1               2               3
// -----------------------------------------------------------------------------
// 1            Uniform         -               -               -
// 2            Uniform         Flat            -               -
// 3            Uniform         Flat            Flat            -
// 4            Uniform         Flat            Flat            Flat
// ...
// DynRank      Uniform         Flat            Flat            Flat
//
// Remove<L>:
//
// Rank \ Index 0               1               2               3
// -----------------------------------------------------------------------------
// 1            Dense           -               -               -
// 2            Flat            Uniform         -               -
// 3            NonUnitStrided  NonUniform      L               -
// 4            NonUnitStrided  NonUniform      NonUniform      L
// ...
// DynRank      NonUnitStrided  NonUniform      NonUniform      NonUniform
//
// Resize<L>:
//
// Rank \ Index 0               1               2               3
// -----------------------------------------------------------------------------
// 1            L               -               -               -
// 2            NonUniform      L               -               -
// 3            NonUniform      NonUniform      L               -
// 4            NonUniform      NonUniform      NonUniform      L
// ...
// DynRank      NonUniform      NonUniform      NonUniform      NonUniform
//

impl Axis for Inner<0> {
    type Dim<S: Shape> = S::Head;
    type Other<S: Shape> = S::Tail;

    type Insert<D: Dim, S: Shape> = S::Prepend<D>;
    type Replace<D: Dim, S: Shape> = <S::Tail as Shape>::Prepend<D>;

    type Keep<S: Shape, L: Layout> = L::Uniform;
    type Remove<S: Shape, L: Layout> = <S::Tail as Shape>::Layout<Flat, L::NonUnitStrided>;
    type Resize<S: Shape, L: Layout> = S::Layout<L, L::NonUniform>;

    fn index(rank: usize) -> usize {
        assert!(rank > 0, "invalid dimension");

        0
    }
}

impl Axis for Inner<1> {
    type Dim<S: Shape> = <S::Tail as Shape>::Head;
    type Other<S: Shape> = <<S::Tail as Shape>::Tail as Shape>::Prepend<S::Head>;

    type Insert<D: Dim, S: Shape> = <<S::Tail as Shape>::Prepend<D> as Shape>::Prepend<S::Head>;
    type Replace<D: Dim, S: Shape> =
        <<<S::Tail as Shape>::Tail as Shape>::Prepend<D> as Shape>::Prepend<S::Head>;

    type Keep<S: Shape, L: Layout> = Flat;
    type Remove<S: Shape, L: Layout> = <S::Tail as Shape>::Layout<L::Uniform, L::NonUniform>;
    type Resize<S: Shape, L: Layout> = <S::Tail as Shape>::Layout<L, L::NonUniform>;

    fn index(rank: usize) -> usize {
        assert!(rank > 1, "invalid dimension");

        1
    }
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*)) => {
        $(
            impl Axis for Inner<$n> {
                type Dim<S: Shape> = <Inner<$k> as Axis>::Dim<S::Tail>;
                type Other<S: Shape> =
                    <<Inner<$k> as Axis>::Other<S::Tail> as Shape>::Prepend<S::Head>;

                type Insert<D: Dim, S: Shape> =
                    <<Inner<$k> as Axis>::Insert<D, S::Tail> as Shape>::Prepend<S::Head>;
                type Replace<D: Dim, S: Shape> =
                    <<Inner<$k> as Axis>::Replace<D, S::Tail> as Shape>::Prepend<S::Head>;

                type Keep<S: Shape, L: Layout> = Flat;
                type Remove<S: Shape, L: Layout> = <Inner<$k> as Axis>::Remove<S::Tail, L>;
                type Resize<S: Shape, L: Layout> = <Inner<$k> as Axis>::Resize<S::Tail, L>;

                fn index(rank: usize) -> usize {
                    assert!(rank > $n, "invalid dimension");

                    $n
                }
            }
        )*
    };
}

impl_axis!((2, 3, 4, 5), (1, 2, 3, 4));

impl Axis for Outer {
    type Dim<S: Shape> = <S::Reverse as Shape>::Head;
    type Other<S: Shape> = <<S::Reverse as Shape>::Tail as Shape>::Reverse;

    type Insert<D: Dim, S: Shape> = <<Inner<0> as Axis>::Insert<D, S::Reverse> as Shape>::Reverse;
    type Replace<D: Dim, S: Shape> = <<Inner<0> as Axis>::Replace<D, S::Reverse> as Shape>::Reverse;

    type Keep<S: Shape, L: Layout> = S::Layout<L::Uniform, Flat>;
    type Remove<S: Shape, L: Layout> = <S::Tail as Shape>::Layout<L::Uniform, L>;
    type Resize<S: Shape, L: Layout> = L;

    fn index(rank: usize) -> usize {
        assert!(rank > 0, "invalid dimension");

        rank - 1
    }
}
