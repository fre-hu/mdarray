use std::slice;

use std::ops::{
    Bound, Index, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::dim::{Const, Dim, Shape, U0, U1};
use crate::format::{Format, Strided};
use crate::layout::{HasLinearIndexing, HasSliceIndexing, Layout, StridedLayout};
use crate::order::{ColumnMajor, Order, RowMajor};
use crate::span::SpanBase;

/// Array index trait for a single dimension.
pub trait DimIndex {
    /// Dimension including the current index.
    type Dim<D: Dim>: Dim;

    /// Format for innermost index.
    type First<F: Format>: Format;

    /// Format for the next index from the innermost index.
    type Root<F: Format>: Format;

    /// Format for indices except the innermost one.
    type Next<F: Format>: Format;

    /// Format for the next index from indices except the innermost one.
    type Outer<F: Format>: Format;

    #[doc(hidden)]
    fn first_dim_info<F: Format, O: Order>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<Self::Dim<U0>, Self::First<F>, O>, Layout<Self::Dim<U0>, Self::Root<F>, O>);

    #[doc(hidden)]
    fn next_dim_info<D: Dim, F: Format, O: Order>(
        self,
        offset: isize,
        inner: Layout<D, F, O>,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<Self::Dim<D>, Self::Next<F>, O>, Layout<Self::Dim<D>, Self::Outer<F>, O>);
}

/// Trait for built-in range types except RangeFull.
pub trait PartialRange<T>: RangeBounds<T> {}

/// Array span index trait for a tuple of indices.
pub trait SpanIndex<D: Dim, O: Order> {
    /// Array span dimension.
    type Dim: Dim;

    /// Format from the inner indices.
    type Inner<F: Format>: Format;

    /// Array span format type.
    type Format<F: Format>: Format;

    /// Format for the next index.
    type Outer<F: Format>: Format;

    #[doc(hidden)]
    fn span_info<F: Format>(
        self,
        layout: Layout<D, F, O>,
    ) -> (isize, Layout<Self::Dim, Self::Format<F>, O>, Layout<Self::Dim, Self::Outer<F>, O>);
}

/// Range constructed from a unit spaced range with the given step size.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepRange<R, S> {
    range: R,
    step: S,
}

/// Returns a range with the given step size from a unit spaced range.
///
/// If the step size is negative, the result is the reverse of the corresponding range
/// with step size as the absolute value of the given step size.
///
/// For example, `step(0..10, 2)` gives the values `0, 2, 4, 6, 8` and `step(0..10, -2)`
/// gives the values `8, 6, 4, 2, 0`.
pub const fn step<R, S>(range: R, step: S) -> StepRange<R, S> {
    StepRange { range, step }
}

impl DimIndex for usize {
    type Dim<D: Dim> = D;

    type First<F: Format> = F;
    type Root<F: Format> = F::NonUnitStrided;

    type Next<F: Format> = F;
    type Outer<F: Format> = F::NonUniform;

    fn first_dim_info<F: Format, O: Order>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<U0, F, O>, Layout<U0, F::NonUnitStrided, O>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (stride * self as isize, Layout::default(), Layout::default())
    }

    fn next_dim_info<D: Dim, F: Format, O: Order>(
        self,
        offset: isize,
        inner: Layout<D, F, O>,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<D, F, O>, Layout<D, F::NonUniform, O>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (offset.wrapping_add(stride * self as isize), inner, inner.reformat())
    }
}

impl DimIndex for RangeFull {
    type Dim<D: Dim> = D::Higher;

    type First<F: Format> = F;
    type Root<F: Format> = F;

    type Next<F: Format> = F;
    type Outer<F: Format> = F;

    fn first_dim_info<F: Format, O: Order>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<U1, F, O>, Layout<U1, F, O>) {
        let layout = Layout::<U0, F, O>::default().add_dim(size, stride);

        (0, layout, layout)
    }

    fn next_dim_info<D: Dim, F: Format, O: Order>(
        self,
        offset: isize,
        inner: Layout<D, F, O>,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<D::Higher, F, O>, Layout<D::Higher, F, O>) {
        let layout = inner.add_dim(size, stride);

        (offset, layout, layout)
    }
}

impl<R: PartialRange<usize>> DimIndex for R {
    type Dim<D: Dim> = D::Higher;

    type First<F: Format> = F;
    type Root<F: Format> = F::NonUniform;

    type Next<F: Format> = F;
    type Outer<F: Format> = F::NonUniform;

    fn first_dim_info<F: Format, O: Order>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<U1, F, O>, Layout<U1, F::NonUniform, O>) {
        let range = slice::range(self, ..size);
        let layout = Layout::<U0, F, O>::default().add_dim(range.end - range.start, stride);

        (stride * range.start as isize, layout, layout.reformat())
    }

    fn next_dim_info<D: Dim, F: Format, O: Order>(
        self,
        offset: isize,
        inner: Layout<D, F, O>,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<D::Higher, F, O>, Layout<D::Higher, F::NonUniform, O>) {
        let range = slice::range(self, ..size);
        let layout = inner.add_dim(range.end - range.start, stride);

        (offset.wrapping_add(stride * range.start as isize), layout, layout.reformat())
    }
}

impl<R: RangeBounds<usize>> DimIndex for StepRange<R, isize> {
    type Dim<D: Dim> = D::Higher;

    type First<F: Format> = F::NonUnitStrided;
    type Root<F: Format> = Strided;

    type Next<F: Format> = F::NonUniform;
    type Outer<F: Format> = F::NonUniform;

    fn first_dim_info<F: Format, O: Order>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<U1, F::NonUnitStrided, O>, StridedLayout<U1, O>) {
        let range = slice::range(self.range, ..size);
        let len = (range.end - range.start).div_ceil(self.step.abs_diff(0));
        let layout = Layout::<U0, F::NonUnitStrided, O>::default().add_dim(len, stride * self.step);

        // Note that the offset may become invalid if the length is zero.
        let delta = if self.step < 0 {
            range.start.wrapping_add(len.wrapping_sub(1).wrapping_mul(self.step.abs_diff(0)))
        } else {
            range.start
        };

        (stride.wrapping_mul(delta as isize), layout, layout.reformat())
    }

    fn next_dim_info<D: Dim, F: Format, O: Order>(
        self,
        offset: isize,
        inner: Layout<D, F, O>,
        size: usize,
        stride: isize,
    ) -> (isize, Layout<D::Higher, F::NonUniform, O>, Layout<D::Higher, F::NonUniform, O>) {
        let range = slice::range(self.range, ..size);
        let len = (range.end - range.start).div_ceil(self.step.abs_diff(0));
        let layout = inner.reformat().add_dim(len, stride * self.step);

        // Note that the offset may become invalid if the length is zero.
        let delta = if self.step < 0 {
            range.start.wrapping_add(len.wrapping_sub(1).wrapping_mul(self.step.abs_diff(0)))
        } else {
            range.start
        };

        (offset.wrapping_add(stride.wrapping_mul(delta as isize)), layout, layout)
    }
}

impl<T> PartialRange<T> for (Bound<T>, Bound<T>) {}
impl<T> PartialRange<T> for Range<T> {}
impl<T> PartialRange<T> for RangeFrom<T> {}
impl<T> PartialRange<T> for RangeInclusive<T> {}
impl<T> PartialRange<T> for RangeTo<T> {}
impl<T> PartialRange<T> for RangeToInclusive<T> {}

impl<O: Order, X: DimIndex> SpanIndex<U1, O> for X {
    type Dim = X::Dim<U0>;

    type Inner<F: Format> = F;
    type Format<F: Format> = X::First<F>;
    type Outer<F: Format> = X::Root<F>;

    fn span_info<F: Format>(
        self,
        layout: Layout<U1, F, O>,
    ) -> (isize, Layout<Self::Dim, Self::Format<F>, O>, Layout<Self::Dim, Self::Outer<F>, O>) {
        self.first_dim_info::<F, O>(layout.size(0), layout.stride(0))
    }
}

macro_rules! impl_view_index_cm {
    ($n:tt, $m:tt, ($($xy:tt),+), $z:tt, ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<$($xy: DimIndex),+, $z: DimIndex> SpanIndex<Const<$n>, ColumnMajor> for ($($xy),+, $z)
        {
            type Dim = $z::Dim<<($($xy),+) as SpanIndex<Const<$m>, ColumnMajor>>::Dim>;

            type Inner<F: Format> = <($($xy),+) as SpanIndex<Const<$m>, ColumnMajor>>::Outer<F>;
            type Format<F: Format> = $z::Next<Self::Inner<F>>;
            type Outer<F: Format> = $z::Outer<Self::Inner<F>>;

            fn span_info<F: Format>(
                self,
                layout: Layout<Const<$n>, F, ColumnMajor>
            ) -> (
                isize,
                Layout<Self::Dim, Self::Format<F>, ColumnMajor>,
                Layout<Self::Dim, Self::Outer<F>, ColumnMajor>
            ) {
                let dim = layout.dim($m);
                let (offset, _, inner) =
                    <($($xy),+) as SpanIndex<Const<$m>, ColumnMajor>>::span_info(
                        ($(self.$idx),+),
                        layout.remove_dim(layout.dim(layout.rank() - 1))
                    );

                self.$m.next_dim_info(offset, inner, layout.size(dim), layout.stride(dim))
            }
        }
    };
}

impl_view_index_cm!(2, 1, (X), Y, (0));
impl_view_index_cm!(3, 2, (X, Y), Z, (0, 1));
impl_view_index_cm!(4, 3, (X, Y, Z), W, (0, 1, 2));
impl_view_index_cm!(5, 4, (X, Y, Z, W), U, (0, 1, 2, 3));
impl_view_index_cm!(6, 5, (X, Y, Z, W, U), V, (0, 1, 2, 3, 4));

macro_rules! impl_view_index_rm {
    ($n:tt, $m:tt, ($($yz:tt),+), ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<X: DimIndex, $($yz: DimIndex),+> SpanIndex<Const<$n>, RowMajor> for (X, $($yz),+)
        {
            type Dim = X::Dim<<($($yz),+) as SpanIndex<Const<$m>, RowMajor>>::Dim>;

            type Inner<F: Format> = <($($yz),+) as SpanIndex<Const<$m>, RowMajor>>::Outer<F>;
            type Format<F: Format> = X::Next<Self::Inner<F>>;
            type Outer<F: Format> = X::Outer<Self::Inner<F>>;

            fn span_info<F: Format>(
                self,
                layout: Layout<Const<$n>, F, RowMajor>
            ) -> (
                isize,
                Layout<Self::Dim, Self::Format<F>, RowMajor>,
                Layout<Self::Dim, Self::Outer<F>, RowMajor>
            ) {
                let dim = layout.dim($m);
                let (offset, _, inner) =
                    <($($yz),+) as SpanIndex<Const<$m>, RowMajor>>::span_info(
                        ($(self.$idx),+),
                        layout.remove_dim(layout.dim(layout.rank() - 1))
                    );

                self.0.next_dim_info(offset, inner, layout.size(dim), layout.stride(dim))
            }
        }
    };
}

impl_view_index_rm!(2, 1, (Y), (1));
impl_view_index_rm!(3, 2, (Y, Z), (1, 2));
impl_view_index_rm!(4, 3, (Y, Z, W), (1, 2, 3));
impl_view_index_rm!(5, 4, (Y, Z, W, U), (1, 2, 3, 4));
impl_view_index_rm!(6, 5, (Y, Z, W, U, V), (1, 2, 3, 4, 5));

impl<T, S: Shape, F: Format, O: Order> Index<S> for SpanBase<T, Layout<S::Dim, F, O>> {
    type Output = T;

    fn index(&self, index: S) -> &T {
        let layout = self.layout();

        for i in 0..self.rank() {
            if index[i] >= layout.size(i) {
                panic_bounds_check(index[i], layout.size(i))
            }
        }

        unsafe { &*self.as_ptr().offset(layout.offset(index)) }
    }
}

impl<T, S: Shape, F: Format, O: Order> IndexMut<S> for SpanBase<T, Layout<S::Dim, F, O>> {
    fn index_mut(&mut self, index: S) -> &mut T {
        let layout = self.layout();

        for i in 0..self.rank() {
            if index[i] >= layout.size(i) {
                panic_bounds_check(index[i], layout.size(i))
            }
        }

        unsafe { &mut *self.as_mut_ptr().offset(layout.offset(index)) }
    }
}

impl<T, D: Dim, F: Format, O: Order> Index<usize> for SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasLinearIndexing,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &*self.as_ptr().offset(self.stride(self.dim(0)) * index as isize) }
    }
}

impl<T, D: Dim, F: Format, O: Order> IndexMut<usize> for SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasLinearIndexing,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &mut *self.as_mut_ptr().offset(self.stride(self.dim(0)) * index as isize) }
    }
}

macro_rules! impl_index_range {
    ($type:ty) => {
        impl<T, D: Dim, F: Format, O: Order> Index<$type> for SpanBase<T, Layout<D, F, O>>
        where
            Layout<D, F, O>: HasSliceIndexing,
        {
            type Output = [T];

            fn index(&self, index: $type) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl<T, D: Dim, F: Format, O: Order> IndexMut<$type> for SpanBase<T, Layout<D, F, O>>
        where
            Layout<D, F, O>: HasSliceIndexing,
        {
            fn index_mut(&mut self, index: $type) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }
    };
}

impl_index_range!((Bound<usize>, Bound<usize>));
impl_index_range!(Range<usize>);
impl_index_range!(RangeFrom<usize>);
impl_index_range!(RangeFull);
impl_index_range!(RangeInclusive<usize>);
impl_index_range!(RangeTo<usize>);
impl_index_range!(RangeToInclusive<usize>);

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}
