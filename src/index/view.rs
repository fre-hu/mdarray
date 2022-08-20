use std::slice;

use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::{Const, Dim, U0, U1};
use crate::format::{Dense, Flat, Format, Uniform};
use crate::layout::{panic_bounds_check, Layout, ViewLayout};
use crate::ops::StepRange;
use crate::order::{ColumnMajor, Order, RowMajor};

/// Helper trait for array indexing, for a single index.
pub trait DimIndex {
    /// Array view parameters.
    type Params<P: Params, F: Format>: Params;

    #[doc(hidden)]
    fn dim_index<P: Params, F: Format, O: Order>(
        self,
        offset: isize,
        layout: ViewLayout<P, O>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewLayout<Self::Params<P, F>, O>);
}

/// Helper trait for array indexing, for view parameters.
pub trait Params {
    /// Array view dimension.
    type Dim: Dim;

    /// Array view format.
    type Format: Format;

    /// Base format for the next larger array view.
    type Larger: Format;

    /// Base format for the next larger array view, with non-uniform stride.
    type Split: Format;
}

/// Array view index trait, for a linear index or a tuple of indices.
pub trait ViewIndex<D: Dim, F: Format, O: Order> {
    /// Array view parameters.
    type Params: Params;

    #[doc(hidden)]
    fn view_index(self, layout: Layout<D, F, O>) -> (isize, ViewLayout<Self::Params, O>);
}

type EmptyParams = (U0, Dense, Dense, Flat);

impl DimIndex for usize {
    type Params<P: Params, F: Format> = (P::Dim, P::Format, P::Split, P::Split);

    fn dim_index<P: Params, F: Format, O: Order>(
        self,
        offset: isize,
        layout: ViewLayout<P, O>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewLayout<Self::Params<P, F>, O>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (offset.wrapping_add(stride * self as isize), layout)
    }
}

impl DimIndex for RangeFull {
    type Params<P: Params, F: Format> = (
        <P::Dim as Dim>::Higher,
        <P::Larger as Format>::Format<<P::Dim as Dim>::Higher, F>,
        P::Larger,
        <P::Larger as Format>::NonUniform,
    );

    fn dim_index<P: Params, F: Format, O: Order>(
        self,
        offset: isize,
        layout: ViewLayout<P, O>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewLayout<Self::Params<P, F>, O>) {
        (offset, layout.add_dim(size, stride))
    }
}

macro_rules! impl_dim_index {
    ($type:ty) => {
        impl DimIndex for $type {
            type Params<P: Params, F: Format> = (
                <P::Dim as Dim>::Higher,
                <P::Larger as Format>::Format<<P::Dim as Dim>::Higher, F>,
                <P::Larger as Format>::NonUniform,
                <P::Larger as Format>::NonUniform,
            );

            fn dim_index<P: Params, F: Format, O: Order>(
                self,
                offset: isize,
                layout: ViewLayout<P, O>,
                size: usize,
                stride: isize,
            ) -> (isize, ViewLayout<Self::Params<P, F>, O>) {
                let range = slice::range(self, ..size);
                let layout = layout.add_dim(range.len(), stride);

                (offset.wrapping_add(stride * range.start as isize), layout)
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
    type Params<P: Params, F: Format> = (
        <P::Dim as Dim>::Higher,
        <P::Split as Format>::Format<<P::Dim as Dim>::Higher, F>,
        <P::Split as Format>::NonUniform,
        <P::Split as Format>::NonUniform,
    );

    fn dim_index<P: Params, F: Format, O: Order>(
        self,
        offset: isize,
        layout: ViewLayout<P, O>,
        size: usize,
        stride: isize,
    ) -> (isize, ViewLayout<Self::Params<P, F>, O>) {
        let range = slice::range(self.range, ..size);
        let len = range.len().div_ceil(self.step.abs_diff(0));
        let layout = layout.add_dim(len, stride * self.step);

        // Note that the offset may become invalid if the length is zero.
        let delta = if self.step < 0 {
            range.start.wrapping_add(len.wrapping_sub(1).wrapping_mul(self.step.abs_diff(0)))
        } else {
            range.start
        };

        (offset.wrapping_add(stride.wrapping_mul(delta as isize)), layout)
    }
}

impl<D: Dim, F: Format, L: Format, S: Format> Params for (D, F, L, S) {
    type Dim = D;
    type Format = F;
    type Larger = L;
    type Split = S;
}

macro_rules! impl_view_index_linear {
    ($type:ty) => {
        impl<D: Dim, F: Uniform, O: Order> ViewIndex<D, F, O> for $type {
            type Params = <Self as DimIndex>::Params<EmptyParams, F>;

            fn view_index(self, layout: Layout<D, F, O>) -> (isize, ViewLayout<Self::Params, O>) {
                let size = layout.len();
                let stride = layout.stride(D::dim::<O>(0));

                self.dim_index::<EmptyParams, F, O>(0, Layout::default(), size, stride)
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

impl<D: Dim, F: Uniform, O: Order, R: RangeBounds<usize>> ViewIndex<D, F, O>
    for StepRange<R, isize>
{
    type Params = <Self as DimIndex>::Params<EmptyParams, F>;

    fn view_index(self, layout: Layout<D, F, O>) -> (isize, ViewLayout<Self::Params, O>) {
        let size = layout.len();
        let stride = layout.stride(D::dim::<O>(0));

        self.dim_index::<EmptyParams, F, O>(0, Layout::default(), size, stride)
    }
}

impl<F: Format, O: Order, X: DimIndex> ViewIndex<U1, F, O> for (X,) {
    type Params = X::Params<EmptyParams, F>;

    fn view_index(self, layout: Layout<U1, F, O>) -> (isize, ViewLayout<Self::Params, O>) {
        let inner = Layout::default();

        self.0.dim_index::<EmptyParams, F, O>(0, inner, layout.size(0), layout.stride(0))
    }
}

macro_rules! impl_view_index {
    ($n:tt, $m:tt, ($($ij:tt),+), $k:tt, ($($xy:tt),+), $z:tt, ($($xyz:tt),+), $f:ty, $o:ty) => {
        impl<F: Format, $($xyz: DimIndex),+> ViewIndex<Const<$n>, F, $o> for ($($xyz),+) {
            type Params = $z::Params<<($($xy),+,) as ViewIndex<Const<$m>, $f, $o>>::Params, F>;

            fn view_index(
                self,
                layout: Layout<Const<$n>, F, $o>,
            ) -> (isize, ViewLayout<Self::Params, $o>) {
                let (offset, inner) = ($(self.$ij),+,).view_index(layout.remove_dim($k));

                self.$k.dim_index(offset, inner, layout.size($k), layout.stride($k))
            }
        }
    };
}

impl_view_index!(2, 1, (0), 1, (X), Y, (X, Y), F::Uniform, ColumnMajor);
impl_view_index!(3, 2, (0, 1), 2, (X, Y), Z, (X, Y, Z), F, ColumnMajor);
impl_view_index!(4, 3, (0, 1, 2), 3, (X, Y, Z), W, (X, Y, Z, W), F, ColumnMajor);
impl_view_index!(5, 4, (0, 1, 2, 3), 4, (X, Y, Z, W), U, (X, Y, Z, W, U), F, ColumnMajor);
impl_view_index!(6, 5, (0, 1, 2, 3, 4), 5, (X, Y, Z, W, U), V, (X, Y, Z, W, U, V), F, ColumnMajor);

impl_view_index!(2, 1, (1), 0, (Y), X, (X, Y), F::Uniform, RowMajor);
impl_view_index!(3, 2, (1, 2), 0, (Y, Z), X, (X, Y, Z), F, RowMajor);
impl_view_index!(4, 3, (1, 2, 3), 0, (Y, Z, W), X, (X, Y, Z, W), F, RowMajor);
impl_view_index!(5, 4, (1, 2, 3, 4), 0, (Y, Z, W, U), X, (X, Y, Z, W, U), F, RowMajor);
impl_view_index!(6, 5, (1, 2, 3, 4, 5), 0, (Y, Z, W, U, V), X, (X, Y, Z, W, U, V), F, RowMajor);
