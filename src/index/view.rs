#[cfg(feature = "nightly")]
use std::slice;

use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::Dyn;
use crate::index;
use crate::layout::{Layout, Strided};
use crate::mapping::Mapping;
use crate::ops::StepRange;
use crate::shape::Shape;

/// Helper trait for array indexing, for a single index.
pub trait DimIndex {
    #[doc(hidden)]
    type Shape<S: Shape, I: ViewIndex>: Shape;

    #[doc(hidden)]
    type Layout<L: Layout, I: ViewIndex>: Layout;

    #[doc(hidden)]
    type Outer<L: Layout, I: ViewIndex>: Layout;

    #[doc(hidden)]
    fn dim_index<S: Shape, M: Mapping, I: ViewIndex, J: ViewIndex<Shape<S> = Self::Shape<S, I>>>(
        self,
        tail: I,
        mapping: &M,
    ) -> (isize, <J::Layout<M::Layout> as Layout>::Mapping<J::Shape<S>>);
}

/// Array view index trait, for a multidimensional index.
pub trait ViewIndex {
    /// Array view shape.
    type Shape<S: Shape>: Shape;

    /// Array view layout.
    type Layout<L: Layout>: Layout;

    #[doc(hidden)]
    type Outer<L: Layout>: Layout;

    #[doc(hidden)]
    const RANK: usize;

    #[doc(hidden)]
    fn view_index<S: Shape, M: Mapping>(
        self,
        mapping: &M,
    ) -> (isize, <Self::Layout<M::Layout> as Layout>::Mapping<Self::Shape<S>>);
}

impl DimIndex for usize {
    type Shape<S: Shape, I: ViewIndex> = I::Shape<S::Tail>;
    type Layout<L: Layout, I: ViewIndex> = I::Layout<L>;
    type Outer<L: Layout, I: ViewIndex> = Strided;

    fn dim_index<S: Shape, M: Mapping, I: ViewIndex, J: ViewIndex<Shape<S> = Self::Shape<S, I>>>(
        self,
        tail: I,
        mapping: &M,
    ) -> (isize, <J::Layout<M::Layout> as Layout>::Mapping<J::Shape<S>>) {
        let (offset, inner) = tail.view_index::<S::Tail, M>(mapping);

        let size = mapping.dim(mapping.rank() - 1 - I::RANK);
        let stride = mapping.stride(mapping.rank() - 1 - I::RANK);

        if self >= size {
            index::panic_bounds_check(self, size)
        }

        (offset + stride * self as isize, Mapping::remap(&inner))
    }
}

impl DimIndex for RangeFull {
    type Shape<S: Shape, I: ViewIndex> = <I::Shape<S::Tail> as Shape>::Prepend<S::Head>;
    type Layout<L: Layout, I: ViewIndex> = I::Outer<L>;
    type Outer<L: Layout, I: ViewIndex> = I::Outer<L>;

    fn dim_index<S: Shape, M: Mapping, I: ViewIndex, J: ViewIndex<Shape<S> = Self::Shape<S, I>>>(
        self,
        tail: I,
        mapping: &M,
    ) -> (isize, <J::Layout<M::Layout> as Layout>::Mapping<J::Shape<S>>) {
        let (offset, inner) = tail.view_index::<S::Tail, M>(mapping);

        let size = mapping.dim(mapping.rank() - 1 - I::RANK);
        let stride = mapping.stride(mapping.rank() - 1 - I::RANK);

        (offset, Mapping::prepend_dim(&inner, size, stride))
    }
}

macro_rules! impl_dim_index {
    ($type:ty) => {
        impl DimIndex for $type {
            type Shape<S: Shape, I: ViewIndex> = <I::Shape<S::Tail> as Shape>::Prepend<Dyn>;
            type Layout<L: Layout, I: ViewIndex> = I::Outer<L>;
            type Outer<L: Layout, I: ViewIndex> = Strided;

            fn dim_index<S: Shape, M: Mapping, I: ViewIndex, J>(
                self,
                tail: I,
                mapping: &M,
            ) -> (isize, <J::Layout<M::Layout> as Layout>::Mapping<J::Shape<S>>)
            where
                J: ViewIndex<Shape<S> = Self::Shape<S, I>>,
            {
                let (offset, inner) = tail.view_index::<S::Tail, M>(mapping);

                let size = mapping.dim(mapping.rank() - 1 - I::RANK);
                let stride = mapping.stride(mapping.rank() - 1 - I::RANK);

                #[cfg(not(feature = "nightly"))]
                let range = crate::index::range(self, ..size);
                #[cfg(feature = "nightly")]
                let range = slice::range(self, ..size);
                let count = stride * range.start as isize;

                (offset + count, Mapping::prepend_dim(&inner, range.len(), stride))
            }
        }
    };
}

impl_dim_index!((Bound<usize>, Bound<usize>));
impl_dim_index!(Range<usize>);
impl_dim_index!(RangeFrom<usize>);
impl_dim_index!(RangeInclusive<usize>);
impl_dim_index!(RangeTo<usize>);
impl_dim_index!(RangeToInclusive<usize>);

impl<R: RangeBounds<usize>> DimIndex for StepRange<R, isize> {
    type Shape<S: Shape, I: ViewIndex> = <I::Shape<S::Tail> as Shape>::Prepend<Dyn>;
    type Layout<L: Layout, I: ViewIndex> = Strided;
    type Outer<L: Layout, I: ViewIndex> = Strided;

    fn dim_index<S: Shape, M: Mapping, I: ViewIndex, J: ViewIndex<Shape<S> = Self::Shape<S, I>>>(
        self,
        tail: I,
        mapping: &M,
    ) -> (isize, <J::Layout<M::Layout> as Layout>::Mapping<J::Shape<S>>) {
        let (offset, inner) = tail.view_index::<S::Tail, M>(mapping);

        let size = mapping.dim(mapping.rank() - 1 - I::RANK);
        let stride = mapping.stride(mapping.rank() - 1 - I::RANK);

        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(self.range, ..size);
        #[cfg(feature = "nightly")]
        let range = slice::range(self.range, ..size);
        let len = range.len().div_ceil(self.step.abs_diff(0));

        let delta = if self.step < 0 && !range.is_empty() { range.end - 1 } else { range.start };

        (offset + stride * delta as isize, Mapping::prepend_dim(&inner, len, stride * self.step))
    }
}

impl ViewIndex for () {
    type Shape<S: Shape> = ();
    type Layout<L: Layout> = L;
    type Outer<L: Layout> = L;

    const RANK: usize = 0;

    fn view_index<S: Shape, M: Mapping>(
        self,
        _: &M,
    ) -> (isize, <Self::Layout<M::Layout> as Layout>::Mapping<Self::Shape<S>>) {
        (0, Default::default())
    }
}

macro_rules! impl_view_index {
    ($n:tt, ($($jk:tt),*), ($($yz:tt),*)) => {
        impl<X: DimIndex $(,$yz: DimIndex)*> ViewIndex for (X, $($yz),*) {
            type Shape<S: Shape> = X::Shape<S, ($($yz,)*)>;
            type Layout<L: Layout> = X::Layout<L, ($($yz,)*)>;
            type Outer<L: Layout> = X::Outer<L, ($($yz,)*)>;

            const RANK: usize = $n;

            fn view_index<S: Shape, M: Mapping>(
                self,
                mapping: &M,
            ) -> (isize, <Self::Layout<M::Layout> as Layout>::Mapping<Self::Shape<S>>) {
                self.0.dim_index::<S, M, ($($yz,)*), Self>(($(self.$jk,)*), mapping)
            }
        }
    };
}

impl_view_index!(1, (), ());
impl_view_index!(2, (1), (Y));
impl_view_index!(3, (1, 2), (Y, Z));
impl_view_index!(4, (1, 2, 3), (Y, Z, W));
impl_view_index!(5, (1, 2, 3, 4), (Y, Z, W, U));
impl_view_index!(6, (1, 2, 3, 4, 5), (Y, Z, W, U, V));
