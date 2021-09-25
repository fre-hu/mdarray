#![allow(unused_parens)]

use crate::order::{ColumnMajor, Order, RowMajor};
use crate::view::{DenseView, StridedView};
use std::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::slice::{self, SliceIndex};

pub enum DimInfo {
    Range(Range<usize>),
    Scalar(usize),
}

pub trait DimIndex: Clone {
    const FULL: bool;
    const RANGE: bool;

    fn dim_info(self, size: usize) -> DimInfo;
}

pub trait IndexMap<const N: usize, O: Order> {
    const FULL: bool;

    const CONT: usize;
    const RANK: usize;

    fn view_info(
        &self,
        dims: &mut [usize],
        shape: &mut [usize],
        start: &mut [usize],
        limit: &[usize],
        dim: usize,
    );
}

pub trait ViewIndex<T, const N: usize, const M: usize, O: Order> {
    type Output: ?Sized;

    fn index(self, view: &StridedView<T, N, M, O>) -> &Self::Output;
    fn index_mut(self, view: &mut StridedView<T, N, M, O>) -> &mut Self::Output;
}

impl DimIndex for usize {
    const FULL: bool = false;
    const RANGE: bool = false;

    fn dim_info(self, limit: usize) -> DimInfo {
        if self >= limit {
            panic_bounds_check(self, limit)
        }

        DimInfo::Scalar(self)
    }
}

macro_rules! impl_dim_index {
    ($type:ty, $full:tt) => {
        impl DimIndex for $type {
            const FULL: bool = $full;
            const RANGE: bool = true;

            fn dim_info(self, limit: usize) -> DimInfo {
                DimInfo::Range(slice::range(self, ..limit))
            }
        }
    };
}

impl_dim_index!((Bound<usize>, Bound<usize>), false);
impl_dim_index!(Range<usize>, false);
impl_dim_index!(RangeFrom<usize>, false);
impl_dim_index!(RangeFull, true);
impl_dim_index!(RangeInclusive<usize>, false);
impl_dim_index!(RangeTo<usize>, false);
impl_dim_index!(RangeToInclusive<usize>, false);

impl<O: Order, X: DimIndex> IndexMap<1, O> for X {
    const FULL: bool = X::FULL;

    const CONT: usize = X::RANGE as usize;
    const RANK: usize = X::RANGE as usize;

    fn view_info(
        &self,
        dims: &mut [usize],
        shape: &mut [usize],
        start: &mut [usize],
        limits: &[usize],
        dim: usize,
    ) {
        start[0] = match self.clone().dim_info(limits[0]) {
            DimInfo::Range(r) => {
                dims[0] = dim;
                shape[0] = r.end - r.start;
                r.start
            }
            DimInfo::Scalar(s) => s,
        };
    }
}

macro_rules! impl_index_map {
    ($n:tt, ($($x:tt),+), ($($y:tt),+), $last:tt, ($($vars:tt),+)) => {
        impl<$($x: DimIndex),+, $last: DimIndex> IndexMap<$n, ColumnMajor> for ($($x),+, $last) {
            const FULL: bool = <($($x),+) as IndexMap<{$n - 1}, ColumnMajor>>::FULL && $last::FULL;

            const CONT: usize = <($($x),+) as IndexMap<{$n - 1}, ColumnMajor>>::CONT
                + (<($($x),+) as IndexMap<{$n - 1}, ColumnMajor>>::FULL && $last::RANGE) as usize;
            const RANK: usize =
                <($($x),+) as IndexMap::<{$n - 1}, ColumnMajor>>::RANK + $last::RANGE as usize;

            fn view_info(&self,
                dims: &mut [usize],
                shape: &mut [usize],
                start: &mut [usize],
                limits: &[usize],
                dim: usize,
            ) {
                start[0] = match self.0.clone().dim_info(limits[0]) {
                    DimInfo::Range(r) => {
                        <($($y),+) as IndexMap<{$n - 1}, ColumnMajor>>::view_info(
                            &($(self.$vars.clone()),+),
                            &mut dims[1..],
                            &mut shape[1..],
                            &mut start[1..],
                            &limits[1..],
                            dim + 1,
                        );
                        dims[0] = dim;
                        shape[0] = r.end - r.start;
                        r.start
                    }
                    DimInfo::Scalar(s) => {
                        <($($y),+) as IndexMap<{$n - 1}, ColumnMajor>>::view_info(
                            &($(self.$vars.clone()),+),
                            dims,
                            shape,
                            &mut start[1..],
                            &limits[1..],
                            dim + 1,
                        );
                        s
                    }
                };
            }
        }

        impl<X: DimIndex, $($y: DimIndex),+> IndexMap<$n, RowMajor> for (X, $($y),+) {
            const FULL: bool = <($($y),+) as IndexMap<{$n - 1}, RowMajor>>::FULL && X::FULL;

            const CONT: usize = <($($y),+) as IndexMap<{$n - 1}, RowMajor>>::CONT
                + (<($($y),+) as IndexMap<{$n - 1}, RowMajor>>::FULL && X::RANGE) as usize;
            const RANK: usize =
                <($($y),+) as IndexMap::<{$n - 1}, RowMajor>>::RANK + X::RANGE as usize;

            fn view_info(&self,
                dims: &mut [usize],
                shape: &mut [usize],
                start: &mut [usize],
                limits: &[usize],
                dim: usize,
            ) {
                start[0] = match self.0.clone().dim_info(limits[0]) {
                    DimInfo::Range(r) => {
                        <($($y),+) as IndexMap<{$n - 1}, RowMajor>>::view_info(
                            &($(self.$vars.clone()),+),
                            &mut dims[1..],
                            &mut shape[1..],
                            &mut start[1..],
                            &limits[1..],
                            dim + 1,
                        );
                        dims[0] = dim;
                        shape[0] = r.end - r.start;
                        r.start
                    }
                    DimInfo::Scalar(s) => {
                        <($($y),+) as IndexMap<{$n - 1}, RowMajor>>::view_info(
                            &($(self.$vars.clone()),+),
                            dims,
                            shape,
                            &mut start[1..],
                            &limits[1..],
                            dim + 1,
                        );
                        s
                    }
                };
            }
        }
    };
}

impl_index_map!(2, (X), (Y), Y, (1));
impl_index_map!(3, (X, Y), (Y, Z), Z, (1, 2));
impl_index_map!(4, (X, Y, Z), (Y, Z, W), W, (1, 2, 3));
impl_index_map!(5, (X, Y, Z, W), (Y, Z, W, U), U, (1, 2, 3, 4));
impl_index_map!(6, (X, Y, Z, W, U), (Y, Z, W, U, V), V, (1, 2, 3, 4, 5));

macro_rules! impl_view_index {
    ($type:ty) => {
        impl<T, const N: usize, O: Order> ViewIndex<T, N, 0, O> for $type {
            type Output = DenseView<T, 1, O>;

            fn index(self, view: &DenseView<T, N, O>) -> &Self::Output {
                <Self as SliceIndex<[T]>>::index(self, view).as_ref()
            }

            fn index_mut(self, view: &mut DenseView<T, N, O>) -> &mut Self::Output {
                <Self as SliceIndex<[T]>>::index_mut(self, view).as_mut()
            }
        }
    };
}

impl_view_index!((Bound<usize>, Bound<usize>));
impl_view_index!(Range<usize>);
impl_view_index!(RangeFrom<usize>);
impl_view_index!(RangeInclusive<usize>);
impl_view_index!(RangeFull);
impl_view_index!(RangeTo<usize>);
impl_view_index!(RangeToInclusive<usize>);

impl<T, const N: usize, O: Order> ViewIndex<T, N, 0, O> for usize {
    type Output = T;

    fn index(self, view: &StridedView<T, N, 0, O>) -> &Self::Output {
        <Self as SliceIndex<[T]>>::index(self, view)
    }

    fn index_mut(self, view: &mut StridedView<T, N, 0, O>) -> &mut Self::Output {
        <Self as SliceIndex<[T]>>::index_mut(self, view)
    }
}

impl<T, const N: usize, const M: usize, O: Order> ViewIndex<T, N, M, O> for [usize; N] {
    type Output = T;

    fn index(self, view: &StridedView<T, N, M, O>) -> &Self::Output {
        let mut index = 0;

        for i in 0..self.len() {
            if self[i] >= view.size(i) {
                panic_bounds_check(self[i], view.size(i))
            }

            index += self[i] as isize * view.stride(i);
        }

        unsafe { &*view.as_ptr().offset(index) }
    }

    fn index_mut(self, view: &mut StridedView<T, N, M, O>) -> &mut Self::Output {
        let mut index = 0;

        for i in 0..self.len() {
            if self[i] >= view.size(i) {
                panic_bounds_check(self[i], view.size(i))
            }

            index += self[i] as isize * view.stride(i);
        }

        unsafe { &mut *view.as_mut_ptr().offset(index) }
    }
}

#[inline(never)]
#[track_caller]
fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!(
        "index out of bounds: the len is {} but the index is {}",
        len, index
    )
}
