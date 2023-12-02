#[cfg(feature = "nightly")]
use std::slice;

use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::{Const, Dim};
use crate::index;
use crate::layout::{Dense, Flat, Layout};
use crate::mapping::{DenseMapping, Mapping};
use crate::ops::StepRange;

/// Helper trait for array indexing, for a single index.
pub trait DimIndex {
    /// Array view dimension.
    type Dim<D: Dim>: Dim;

    /// Array view parameters.
    type Params<P: Params>: Params;

    #[doc(hidden)]
    fn dim_index<D: Dim, L: Layout, I, O>(
        self,
        index: I,
        mapping: L::Mapping<D>,
    ) -> (isize, <O::Layout as Layout>::Mapping<O::Dim>)
    where
        I: ViewIndex<D::Lower, <D::Lower as Dim>::Layout<L>>,
        O: ViewIndex<D, L, Dim = Self::Dim<I::Dim>>;
}

/// Helper trait for array indexing, for view parameters.
pub trait Params {
    /// Base layout for the array view.
    type Layout: Layout;

    /// Base layout for the parent array view.
    type Parent: Layout;

    /// Base layout for the parent array view, when splitting along the dimension.
    type Split: Layout;
}

/// Array view index trait, for a linear index or a tuple of indices.
pub trait ViewIndex<D: Dim, L: Layout> {
    /// Array view dimension.
    type Dim: Dim;

    /// Array view parameters.
    type Params: Params;

    /// Array view layout.
    #[cfg(not(feature = "nightly"))]
    type Layout: Layout;

    /// Array view layout.
    #[cfg(feature = "nightly")]
    type Layout: Layout = <<Self::Params as Params>::Layout as Layout>::Layout<Self::Dim, L>;

    #[doc(hidden)]
    fn view_index(
        self,
        mapping: L::Mapping<D>,
    ) -> (isize, <Self::Layout as Layout>::Mapping<Self::Dim>);
}

impl DimIndex for usize {
    type Dim<D: Dim> = D;
    type Params<P: Params> = (P::Layout, P::Split, P::Split);

    fn dim_index<D: Dim, L: Layout, I, O>(
        self,
        index: I,
        mapping: L::Mapping<D>,
    ) -> (isize, <O::Layout as Layout>::Mapping<O::Dim>)
    where
        I: ViewIndex<D::Lower, <D::Lower as Dim>::Layout<L>>,
        O: ViewIndex<D, L, Dim = Self::Dim<I::Dim>>,
    {
        let (offset, inner) = index.view_index(Mapping::remove_dim(mapping, D::RANK - 1));

        let size = mapping.size(D::RANK - 1);
        let stride = mapping.stride(D::RANK - 1);

        if self >= size {
            index::panic_bounds_check(self, size)
        }

        (offset + stride * self as isize, Mapping::remap(inner))
    }
}

impl DimIndex for RangeFull {
    type Dim<D: Dim> = D::Higher;
    type Params<P: Params> = (P::Parent, P::Parent, <P::Parent as Layout>::NonUniform);

    fn dim_index<D: Dim, L: Layout, I, O>(
        self,
        index: I,
        mapping: L::Mapping<D>,
    ) -> (isize, <O::Layout as Layout>::Mapping<O::Dim>)
    where
        I: ViewIndex<D::Lower, <D::Lower as Dim>::Layout<L>>,
        O: ViewIndex<D, L, Dim = Self::Dim<I::Dim>>,
    {
        let (offset, inner) = index.view_index(Mapping::remove_dim(mapping, D::RANK - 1));

        let size = mapping.size(D::RANK - 1);
        let stride = mapping.stride(D::RANK - 1);

        (offset, Mapping::add_dim(inner, size, stride))
    }
}

macro_rules! impl_dim_index {
    ($type:ty) => {
        impl DimIndex for $type {
            type Dim<D: Dim> = D::Higher;
            type Params<P: Params> =
                (P::Parent, <P::Parent as Layout>::NonUniform, <P::Parent as Layout>::NonUniform);

            fn dim_index<D: Dim, L: Layout, I, O>(
                self,
                index: I,
                mapping: L::Mapping<D>,
            ) -> (isize, <O::Layout as Layout>::Mapping<O::Dim>)
            where
                I: ViewIndex<D::Lower, <D::Lower as Dim>::Layout<L>>,
                O: ViewIndex<D, L, Dim = Self::Dim<I::Dim>>,
            {
                let (offset, inner) = index.view_index(Mapping::remove_dim(mapping, D::RANK - 1));

                let size = mapping.size(D::RANK - 1);
                let stride = mapping.stride(D::RANK - 1);

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
    type Dim<D: Dim> = D::Higher;
    type Params<P: Params> =
        (P::Split, <P::Split as Layout>::NonUniform, <P::Split as Layout>::NonUniform);

    fn dim_index<D: Dim, L: Layout, I, O>(
        self,
        index: I,
        mapping: L::Mapping<D>,
    ) -> (isize, <O::Layout as Layout>::Mapping<O::Dim>)
    where
        I: ViewIndex<D::Lower, <D::Lower as Dim>::Layout<L>>,
        O: ViewIndex<D, L, Dim = Self::Dim<I::Dim>>,
    {
        let (offset, inner) = index.view_index(Mapping::remove_dim(mapping, D::RANK - 1));

        let size = mapping.size(D::RANK - 1);
        let stride = mapping.stride(D::RANK - 1);

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

impl<L: Layout, P: Layout, S: Layout> Params for (L, P, S) {
    type Layout = L;
    type Parent = P;
    type Split = S;
}

impl ViewIndex<Const<0>, Dense> for () {
    type Dim = Const<0>;
    type Params = (Dense, Dense, Flat);

    #[cfg(not(feature = "nightly"))]
    type Layout = <<Self::Params as Params>::Layout as Layout>::Layout<Self::Dim, Dense>;

    fn view_index(self, _: DenseMapping<Const<0>>) -> (isize, DenseMapping<Const<0>>) {
        (0, DenseMapping::default())
    }
}

macro_rules! impl_view_index {
    ($n:tt, $m:tt, ($($ij:tt),*), $k:tt, ($($xy:tt),*), $z:tt, ($($xyz:tt),+), $layout:ty) => {
        impl<L: Layout, $($xyz: DimIndex),+> ViewIndex<Const<$n>, L> for ($($xyz,)+) {
            type Dim = $z::Dim<<($($xy,)*) as ViewIndex<Const<$m>, $layout>>::Dim>;
            type Params = $z::Params<<($($xy,)*) as ViewIndex<Const<$m>, $layout>>::Params>;

            #[cfg(not(feature = "nightly"))]
            type Layout = <<Self::Params as Params>::Layout as Layout>::Layout<Self::Dim, L>;

            fn view_index(
                self,
                mapping: L::Mapping<Const<$n>>,
            ) -> (isize, <Self::Layout as Layout>::Mapping<Self::Dim>) {
                self.$k.dim_index::<Const<$n>, L, ($($xy,)*), Self>(($(self.$ij,)*), mapping)
            }
        }
    };
}

impl_view_index!(1, 0, (), 0, (), X, (X), Dense);
impl_view_index!(2, 1, (0), 1, (X), Y, (X, Y), L::Uniform);
impl_view_index!(3, 2, (0, 1), 2, (X, Y), Z, (X, Y, Z), L);
impl_view_index!(4, 3, (0, 1, 2), 3, (X, Y, Z), W, (X, Y, Z, W), L);
impl_view_index!(5, 4, (0, 1, 2, 3), 4, (X, Y, Z, W), U, (X, Y, Z, W, U), L);
impl_view_index!(6, 5, (0, 1, 2, 3, 4), 5, (X, Y, Z, W, U), V, (X, Y, Z, W, U, V), L);
