#[cfg(feature = "nightly")]
use std::slice;

use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::{Const, Dim};
use crate::index::panic_bounds_check;
use crate::layout::{Dense, Flat, Layout, Uniform};
use crate::mapping::{DenseMapping, Mapping};
use crate::ops::StepRange;

/// Helper trait for array indexing, for a single index.
pub trait DimIndex {
    /// Array view parameters.
    type Params<P: Params, L: Layout>: Params;

    #[doc(hidden)]
    fn dim_index<P: Params, L: Layout>(
        self,
        offset: isize,
        mapping: ViewMapping<P>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewMapping<Self::Params<P, L>>);
}

/// Helper trait for array indexing, for view parameters.
pub trait Params {
    /// Array view dimension.
    type Dim: Dim;

    /// Array view layout.
    type Layout: Layout;

    /// Base layout for the next larger array view.
    type Larger: Layout;

    /// Base layout for the next larger array view, with non-uniform stride.
    type Split: Layout;
}

/// Array view index trait, for a linear index or a tuple of indices.
pub trait ViewIndex<D: Dim, L: Layout> {
    /// Array view parameters.
    type Params: Params;

    #[doc(hidden)]
    fn view_index(self, mapping: L::Mapping<D>) -> (isize, ViewMapping<Self::Params>);
}

type EmptyParams = (Const<0>, Dense, Dense, Flat);

type ViewMapping<P> = <<P as Params>::Layout as Layout>::Mapping<<P as Params>::Dim>;

impl DimIndex for usize {
    type Params<P: Params, L: Layout> = (P::Dim, P::Layout, P::Split, P::Split);

    fn dim_index<P: Params, L: Layout>(
        self,
        offset: isize,
        mapping: ViewMapping<P>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewMapping<Self::Params<P, L>>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (offset.wrapping_add(stride * self as isize), mapping)
    }
}

impl DimIndex for RangeFull {
    type Params<P: Params, L: Layout> = (
        <P::Dim as Dim>::Higher,
        <P::Larger as Layout>::Layout<<P::Dim as Dim>::Higher, L>,
        P::Larger,
        <P::Larger as Layout>::NonUniform,
    );

    fn dim_index<P: Params, L: Layout>(
        self,
        offset: isize,
        mapping: ViewMapping<P>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewMapping<Self::Params<P, L>>) {
        (offset, Mapping::add_dim(mapping, size, stride))
    }
}

macro_rules! impl_dim_index {
    ($type:ty) => {
        impl DimIndex for $type {
            type Params<P: Params, L: Layout> = (
                <P::Dim as Dim>::Higher,
                <P::Larger as Layout>::Layout<<P::Dim as Dim>::Higher, L>,
                <P::Larger as Layout>::NonUniform,
                <P::Larger as Layout>::NonUniform,
            );

            fn dim_index<P: Params, L: Layout>(
                self,
                offset: isize,
                mapping: ViewMapping<P>,
                size: usize,
                stride: isize,
            ) -> (isize, ViewMapping<Self::Params<P, L>>) {
                #[cfg(not(feature = "nightly"))]
                let range = crate::index::range(self, ..size);
                #[cfg(feature = "nightly")]
                let range = slice::range(self, ..size);
                let mapping = Mapping::add_dim(mapping, range.len(), stride);

                (offset.wrapping_add(stride * range.start as isize), mapping)
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
    type Params<P: Params, L: Layout> = (
        <P::Dim as Dim>::Higher,
        <P::Split as Layout>::Layout<<P::Dim as Dim>::Higher, L>,
        <P::Split as Layout>::NonUniform,
        <P::Split as Layout>::NonUniform,
    );

    fn dim_index<P: Params, L: Layout>(
        self,
        offset: isize,
        mapping: ViewMapping<P>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewMapping<Self::Params<P, L>>) {
        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(self.range, ..size);
        #[cfg(feature = "nightly")]
        let range = slice::range(self.range, ..size);
        #[cfg(not(feature = "nightly"))]
        let len = crate::index::div_ceil(range.len(), self.step.abs_diff(0));
        #[cfg(feature = "nightly")]
        let len = range.len().div_ceil(self.step.abs_diff(0));
        let mapping = Mapping::add_dim(mapping, len, stride * self.step);

        // Note that the offset may become invalid if the length is zero.
        let delta = if self.step < 0 {
            range.start.wrapping_add(len.wrapping_sub(1).wrapping_mul(self.step.abs_diff(0)))
        } else {
            range.start
        };

        (offset.wrapping_add(stride.wrapping_mul(delta as isize)), mapping)
    }
}

impl<D: Dim, L: Layout, M: Layout, S: Layout> Params for (D, L, M, S) {
    type Dim = D;
    type Layout = L;
    type Larger = M;
    type Split = S;
}

macro_rules! impl_view_index_linear {
    ($type:ty) => {
        impl<D: Dim, L: Uniform> ViewIndex<D, L> for $type {
            type Params = <Self as DimIndex>::Params<EmptyParams, L>;

            fn view_index(self, mapping: L::Mapping<D>) -> (isize, ViewMapping<Self::Params>) {
                let inner = DenseMapping::default();

                self.dim_index::<EmptyParams, L>(0, inner, mapping.size(0), mapping.stride(0))
            }
        }
    };
}

impl_view_index_linear!(usize);
impl_view_index_linear!((Bound<usize>, Bound<usize>));
impl_view_index_linear!(Range<usize>);
impl_view_index_linear!(RangeFrom<usize>);
impl_view_index_linear!(RangeFull);
impl_view_index_linear!(RangeInclusive<usize>);
impl_view_index_linear!(RangeTo<usize>);
impl_view_index_linear!(RangeToInclusive<usize>);

impl<D: Dim, L: Uniform, R: RangeBounds<usize>> ViewIndex<D, L> for StepRange<R, isize> {
    type Params = <Self as DimIndex>::Params<EmptyParams, L>;

    fn view_index(self, mapping: L::Mapping<D>) -> (isize, ViewMapping<Self::Params>) {
        let inner = DenseMapping::default();

        self.dim_index::<EmptyParams, L>(0, inner, mapping.size(0), mapping.stride(0))
    }
}

impl<L: Layout, X: DimIndex> ViewIndex<Const<1>, L> for (X,) {
    type Params = X::Params<EmptyParams, L>;

    fn view_index(self, mapping: L::Mapping<Const<1>>) -> (isize, ViewMapping<Self::Params>) {
        let inner = DenseMapping::default();

        self.0.dim_index::<EmptyParams, L>(0, inner, mapping.size(0), mapping.stride(0))
    }
}

macro_rules! impl_view_index {
    ($n:tt, $m:tt, ($($ij:tt),+), $k:tt, ($($xy:tt),+), $z:tt, ($($xyz:tt),+), $f:ty) => {
        impl<L: Layout, $($xyz: DimIndex),+> ViewIndex<Const<$n>, L> for ($($xyz),+) {
            type Params = $z::Params<<($($xy),+,) as ViewIndex<Const<$m>, $f>>::Params, L>;

            fn view_index(
                self,
                mapping: L::Mapping<Const<$n>>,
            ) -> (isize, ViewMapping<Self::Params>) {
                let (offset, inner) = ($(self.$ij),+,).view_index(Mapping::remove_dim(mapping, $k));

                self.$k.dim_index(offset, inner, mapping.size($k), mapping.stride($k))
            }
        }
    };
}

impl_view_index!(2, 1, (0), 1, (X), Y, (X, Y), L::Uniform);
impl_view_index!(3, 2, (0, 1), 2, (X, Y), Z, (X, Y, Z), L);
impl_view_index!(4, 3, (0, 1, 2), 3, (X, Y, Z), W, (X, Y, Z, W), L);
impl_view_index!(5, 4, (0, 1, 2, 3), 4, (X, Y, Z, W), U, (X, Y, Z, W, U), L);
impl_view_index!(6, 5, (0, 1, 2, 3, 4), 5, (X, Y, Z, W, U), V, (X, Y, Z, W, U, V), L);
