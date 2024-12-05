use std::fmt::Debug;
use std::hash::Hash;

use crate::dim::{Const, Dim, Dyn};
use crate::layout::Layout;
use crate::mapping::{DenseMapping, Mapping};
use crate::shape::{DynRank, Shape};

/// Array axis trait, for subarray shapes.
pub trait Axis: Copy + Debug + Default + Eq + Hash + Ord + Send + Sync {
    /// Corresponding dimension.
    type Dim<S: Shape>: Dim;

    /// Shape for the other dimensions.
    type Other<S: Shape>: Shape;

    /// Shape for the previous dimensions excluding the current dimension.
    type Init<S: Shape>: Shape;

    /// Shape for the next dimensions including the current dimension.
    type Rest<S: Shape>: Shape;

    /// Replace the dimension in the shape.
    type Replace<D: Dim, S: Shape>: Shape;

    #[doc(hidden)]
    fn index(self, rank: usize) -> usize;

    #[doc(hidden)]
    fn keep<M: Mapping>(
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
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Self::Other<M::Shape>> {
        Mapping::remove_dim::<M>(mapping, self.index(mapping.rank()))
    }

    #[doc(hidden)]
    fn resize<M: Mapping>(
        self,
        mapping: &M,
        new_size: usize,
    ) -> <Split<Self, M::Shape, M::Layout> as Layout>::Mapping<Self::Replace<Dyn, M::Shape>> {
        Mapping::resize_dim::<M>(mapping, self.index(mapping.rank()), new_size)
    }
}

/// Layout when keeping the dimension for the specified axis.
pub type Keep<A, S, L> = <<<A as Axis>::Rest<S> as Shape>::Tail as Shape>::Layout<L>;

/// Layout when removing or resizing the dimension for the specified axis.
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
    type Other<S: Shape> = S::Tail;

    type Init<S: Shape> = ();
    type Rest<S: Shape> = S;

    type Replace<D: Dim, S: Shape> = <S::Tail as Shape>::Prepend<D>;

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
                type Other<S: Shape> =
                    <<Const<$k> as Axis>::Other<S::Tail> as Shape>::Prepend<S::Head>;

                type Init<S: Shape> =
                    <<Const<$k> as Axis>::Init<S::Tail> as Shape>::Prepend<S::Head>;
                type Rest<S: Shape> = <Const<$k> as Axis>::Rest<S::Tail>;

                type Replace<D: Dim, S: Shape> =
                    <<Const<$k> as Axis>::Replace<D, S::Tail> as Shape>::Prepend<S::Head>;

                fn index(self, rank: usize) -> usize {
                    assert!(rank > $n, "invalid dimension");

                    $n
                }
            }
        )*
    };
}

impl_axis!((1, 2, 3, 4, 5), (0, 1, 2, 3, 4));

impl Axis for Dyn {
    type Dim<S: Shape> = Dyn;
    type Other<S: Shape> = <S::Tail as Shape>::Dyn;

    type Init<S: Shape> = DynRank;
    type Rest<S: Shape> = DynRank;

    type Replace<D: Dim, S: Shape> = S::Dyn;

    fn index(self, rank: usize) -> usize {
        assert!(self < rank, "invalid dimension");

        self
    }
}
