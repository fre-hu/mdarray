#[cfg(feature = "nightly")]
use std::slice;

use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::Dyn;
use crate::index::{self, Axis, Outer};
use crate::layout::{Dense, Flat, Layout};
use crate::mapping::Mapping;
use crate::ops::StepRange;
use crate::shape::Shape;

/// Helper trait for array indexing, for a single index.
pub trait DimIndex {
    /// Array view shape.
    type Shape<S: Shape, I: ViewIndex>: Shape;

    /// Array view parameters.
    type Params<P: Params>: Params;

    #[doc(hidden)]
    fn dim_index<M: Mapping, I: ViewIndex, J>(
        self,
        index: I,
        mapping: M,
    ) -> (isize, <J::Layout<M::Shape, M::Layout> as Layout>::Mapping<J::Shape<M::Shape>>)
    where
        J: ViewIndex<Shape<M::Shape> = Self::Shape<M::Shape, I>>;
}

/// Helper trait for array indexing, for view parameters.
pub trait Params {
    /// Base layout for the array view.
    type Layout: Layout;

    /// Base layout for the outer array view.
    type Outer: Layout;

    /// Base layout for the outer array view, when splitting along the dimension.
    type Split: Layout;
}

/// Array view index trait, for a tuple of indices.
pub trait ViewIndex {
    /// Array view shape.
    type Shape<S: Shape>: Shape;

    /// Array view layout.
    type Layout<S: Shape, L: Layout>: Layout;

    /// Array view parameters.
    type Params: Params;

    #[doc(hidden)]
    fn view_index<M: Mapping>(
        self,
        mapping: M,
    ) -> (isize, <Self::Layout<M::Shape, M::Layout> as Layout>::Mapping<Self::Shape<M::Shape>>);
}

type Merge<P, L> = <<P as Params>::Layout as Layout>::Merge<L>;

impl DimIndex for usize {
    type Shape<S: Shape, I: ViewIndex> = I::Shape<<Outer as Axis>::Other<S>>;
    type Params<P: Params> = (P::Layout, P::Split, P::Split);

    fn dim_index<M: Mapping, I: ViewIndex, J>(
        self,
        index: I,
        mapping: M,
    ) -> (isize, <J::Layout<M::Shape, M::Layout> as Layout>::Mapping<J::Shape<M::Shape>>)
    where
        J: ViewIndex<Shape<M::Shape> = Self::Shape<M::Shape, I>>,
    {
        let (offset, inner) = index.view_index(Outer::remove(mapping));

        let size = mapping.dim(M::Shape::RANK - 1);
        let stride = mapping.stride(M::Shape::RANK - 1);

        if self >= size {
            index::panic_bounds_check(self, size)
        }

        (offset + stride * self as isize, Mapping::remap(inner))
    }
}

impl DimIndex for RangeFull {
    type Shape<S: Shape, I: ViewIndex> =
        <Outer as Axis>::Insert<<Outer as Axis>::Dim<S>, I::Shape<<Outer as Axis>::Other<S>>>;
    type Params<P: Params> = (P::Outer, P::Outer, <P::Outer as Layout>::NonUniform);

    fn dim_index<M: Mapping, I: ViewIndex, J>(
        self,
        index: I,
        mapping: M,
    ) -> (isize, <J::Layout<M::Shape, M::Layout> as Layout>::Mapping<J::Shape<M::Shape>>)
    where
        J: ViewIndex<Shape<M::Shape> = Self::Shape<M::Shape, I>>,
    {
        let (offset, inner) = index.view_index(Outer::remove(mapping));

        let size = mapping.dim(M::Shape::RANK - 1);
        let stride = mapping.stride(M::Shape::RANK - 1);

        (offset, Mapping::add_dim(inner, size, stride))
    }
}

macro_rules! impl_dim_index {
    ($type:ty) => {
        impl DimIndex for $type {
            type Shape<S: Shape, I: ViewIndex> =
                <Outer as Axis>::Insert<Dyn, I::Shape<<Outer as Axis>::Other<S>>>;
            type Params<P: Params> =
                (P::Outer, <P::Outer as Layout>::NonUniform, <P::Outer as Layout>::NonUniform);

            fn dim_index<M: Mapping, I: ViewIndex, J>(
                self,
                index: I,
                mapping: M,
            ) -> (isize, <J::Layout<M::Shape, M::Layout> as Layout>::Mapping<J::Shape<M::Shape>>)
            where
                J: ViewIndex<Shape<M::Shape> = Self::Shape<M::Shape, I>>,
            {
                let (offset, inner) = index.view_index(Outer::remove(mapping));

                let size = mapping.dim(M::Shape::RANK - 1);
                let stride = mapping.stride(M::Shape::RANK - 1);

                #[cfg(not(feature = "nightly"))]
                let range = crate::index::range(self, ..size);
                #[cfg(feature = "nightly")]
                let range = slice::range(self, ..size);
                let count = stride * range.start as isize;

                (offset + count, Mapping::add_dim(inner, range.len(), stride))
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
    type Shape<S: Shape, I: ViewIndex> =
        <Outer as Axis>::Insert<Dyn, I::Shape<<Outer as Axis>::Other<S>>>;
    type Params<P: Params> =
        (P::Split, <P::Split as Layout>::NonUniform, <P::Split as Layout>::NonUniform);

    fn dim_index<M: Mapping, I: ViewIndex, J>(
        self,
        index: I,
        mapping: M,
    ) -> (isize, <J::Layout<M::Shape, M::Layout> as Layout>::Mapping<J::Shape<M::Shape>>)
    where
        J: ViewIndex<Shape<M::Shape> = Self::Shape<M::Shape, I>>,
    {
        let (offset, inner) = index.view_index(Outer::remove(mapping));

        let size = mapping.dim(M::Shape::RANK - 1);
        let stride = mapping.stride(M::Shape::RANK - 1);

        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(self.range, ..size);
        #[cfg(feature = "nightly")]
        let range = slice::range(self.range, ..size);
        #[cfg(not(feature = "nightly"))]
        let len = crate::index::div_ceil(range.len(), self.step.abs_diff(0));
        #[cfg(feature = "nightly")]
        let len = range.len().div_ceil(self.step.abs_diff(0));

        let delta = if self.step < 0 && !range.is_empty() { range.end - 1 } else { range.start };

        (offset + stride * delta as isize, Mapping::add_dim(inner, len, stride * self.step))
    }
}

impl<L: Layout, O: Layout, S: Layout> Params for (L, O, S) {
    type Layout = L;
    type Outer = O;
    type Split = S;
}

impl ViewIndex for () {
    type Shape<S: Shape> = ();
    type Layout<S: Shape, L: Layout> = Dense;
    type Params = (Dense, Dense, Flat);

    fn view_index<M: Mapping>(
        self,
        _: M,
    ) -> (isize, <Self::Layout<M::Shape, M::Layout> as Layout>::Mapping<Self::Shape<M::Shape>>)
    {
        (0, Default::default())
    }
}

macro_rules! impl_view_index {
    ($n:tt, ($($ij:tt),*), $k:tt, ($($xy:tt),*), $z:tt) => {
        impl<$($xy: DimIndex,)* $z: DimIndex> ViewIndex for ($($xy,)* $z,) {
            type Shape<S: Shape> = $z::Shape<S, ($($xy,)*)>;
            type Layout<S: Shape, L: Layout> =
                Merge<Self::Params, <Self::Shape<S> as Shape>::Layout<L::Uniform, L>>;
            type Params = $z::Params<<($($xy,)*) as ViewIndex>::Params>;

            fn view_index<M: Mapping>(
                self,
                mapping: M,
            ) -> (
                isize,
                <Self::Layout<M::Shape, M::Layout> as Layout>::Mapping<Self::Shape<M::Shape>>
            ) {
                self.$k.dim_index::<M, ($($xy,)*), Self>(($(self.$ij,)*), mapping)
            }
        }
    };
}

impl_view_index!(1, (), 0, (), X);
impl_view_index!(2, (0), 1, (X), Y);
impl_view_index!(3, (0, 1), 2, (X, Y), Z);
impl_view_index!(4, (0, 1, 2), 3, (X, Y, Z), W);
impl_view_index!(5, (0, 1, 2, 3), 4, (X, Y, Z, W), U);
impl_view_index!(6, (0, 1, 2, 3, 4), 5, (X, Y, Z, W, U), V);
