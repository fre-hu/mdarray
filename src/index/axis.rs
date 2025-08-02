use core::fmt::Debug;
use core::hash::Hash;

use crate::dim::{Const, Dim, Dyn};
use crate::layout::Layout;
use crate::mapping::{DenseMapping, Mapping};
use crate::shape::{DynRank, Shape};

/// Array axis trait, for subarray shapes.
pub trait Axis: Copy + Debug + Default + Hash + Ord + Send + Sync {
    /// Corresponding dimension.
    type Dim<S: Shape>: Dim;

    /// Shape for the previous dimensions excluding the current dimension.
    type Init<S: Shape>: Shape;

    /// Shape for the next dimensions excluding the current dimension.
    type Rest<S: Shape>: Shape;

    /// Remove the dimension from the shape.
    type Remove<S: Shape>: Shape;

    /// Insert the dimension into the shape.
    type Insert<D: Dim, S: Shape>: Shape;

    /// Returns the dimension index.
    fn index(self, rank: usize) -> usize;

    #[doc(hidden)]
    fn get<M: Mapping>(
        self,
        mapping: &M,
    ) -> <Keep<Self, M::Shape, M::Layout> as Layout>::Mapping<(Self::Dim<M::Shape>,)> {
        let index = self.index(mapping.rank());

        Mapping::prepend_dim(&DenseMapping::new(()), mapping.dim(index), mapping.stride(index))
    }

    #[doc(hidden)]
    fn remove<M: Mapping>(
        self,
        mapping: &M,
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Self::Remove<M::Shape>> {
        Mapping::remove_dim::<M>(mapping, self.index(mapping.rank()))
    }

    #[doc(hidden)]
    fn resize<M: Mapping>(
        self,
        mapping: &M,
        new_size: usize,
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Resize<Self, M::Shape>> {
        Mapping::resize_dim::<M>(mapping, self.index(mapping.rank()), new_size)
    }
}

/// Column axis type, for the second last dimension.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Cols;

/// Row axis type, for the last dimension.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Rows;

//
// These types are public to improve documentation, but hidden since
// they are not considered part of the API.
//

#[doc(hidden)]
pub type Resize<A, S> = <A as Axis>::Insert<Dyn, <A as Axis>::Remove<S>>;

#[doc(hidden)]
pub type Keep<A, S, L> = <<A as Axis>::Rest<S> as Shape>::Layout<L>;

#[doc(hidden)]
pub type Split<A, S, L> = <<A as Axis>::Init<S> as Shape>::Layout<L>;

//
// The tables below give the resulting layout depending on the rank and axis.
//
// Keep<A, S, L>:
//
// Rank \ Axis  0           1           2           ...         Dyn
// -------------------------------------------------------------------------
// 1            L           -           -           -           Strided
// 2            Strided     L           -           -           Strided
// 3            Strided     Strided     L           -           Strided
// ...
// DynRank      Strided     Strided     Strided     ...         Strided
//
// Split<A, S, L>:
//
// Rank \ Axis  0           1           2           ...         Dyn
// -------------------------------------------------------------------------
// 1            L           -           -           -           Strided
// 2            L           Strided     -           -           Strided
// 3            L           Strided     Strided     -           Strided
// ...
// DynRank      L           Strided     Strided     ...         Strided
//

impl Axis for Const<0> {
    type Dim<S: Shape> = S::Head;

    type Init<S: Shape> = ();
    type Rest<S: Shape> = S::Tail;

    type Remove<S: Shape> = S::Tail;
    type Insert<D: Dim, S: Shape> = S::Prepend<D>;

    fn index(self, rank: usize) -> usize {
        assert!(rank > 0, "invalid dimension");

        0
    }
}

macro_rules! impl_axis {
    (($($n:tt),*), ($($k:tt),*)) => {
        $(
            impl Axis for Const<$n> {
                type Dim<S: Shape> = <Const<$k> as Axis>::Dim<S::Tail>;

                type Init<S: Shape> =
                    <<Const<$k> as Axis>::Init<S::Tail> as Shape>::Prepend<S::Head>;
                type Rest<S: Shape> = <Const<$k> as Axis>::Rest<S::Tail>;

                type Remove<S: Shape> =
                    <<Const<$k> as Axis>::Remove<S::Tail> as Shape>::Prepend<S::Head>;
                type Insert<D: Dim, S: Shape> =
                    <<Const<$k> as Axis>::Insert<D, S::Tail> as Shape>::Prepend<S::Head>;

                fn index(self, rank: usize) -> usize {
                    assert!(rank > $n, "invalid dimension");

                    $n
                }
            }
        )*
    };
}

impl_axis!((1, 2, 3, 4, 5), (0, 1, 2, 3, 4));

macro_rules! impl_cols_rows {
    ($name:tt, $n:tt) => {
        impl Axis for $name {
            type Dim<S: Shape> = <Const<$n> as Axis>::Dim<S::Reverse>;

            type Init<S: Shape> = <<Const<$n> as Axis>::Rest<S::Reverse> as Shape>::Reverse;
            type Rest<S: Shape> = <<Const<$n> as Axis>::Init<S::Reverse> as Shape>::Reverse;

            type Remove<S: Shape> = <<Const<$n> as Axis>::Remove<S::Reverse> as Shape>::Reverse;
            type Insert<D: Dim, S: Shape> =
                <<Const<$n> as Axis>::Insert<D, S::Reverse> as Shape>::Reverse;

            fn index(self, rank: usize) -> usize {
                assert!(rank > $n, "invalid dimension");

                rank - $n - 1
            }
        }
    };
}

impl_cols_rows!(Cols, 1);
impl_cols_rows!(Rows, 0);

impl Axis for Dyn {
    type Dim<S: Shape> = Dyn;

    type Init<S: Shape> = DynRank;
    type Rest<S: Shape> = DynRank;

    type Remove<S: Shape> = <S::Tail as Shape>::Dyn;
    type Insert<D: Dim, S: Shape> = <S::Dyn as Shape>::Prepend<Dyn>;

    fn index(self, rank: usize) -> usize {
        assert!(self < rank, "invalid dimension");

        self
    }
}
