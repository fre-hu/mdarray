use std::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::{Dim, Shape};
use crate::format::{Format, Uniform};
use crate::layout::{panic_bounds_check, Layout};
use crate::order::Order;
use crate::span::{DenseSpan, SpanBase};

impl<T, S: Shape, F: Format, O: Order> Index<S> for SpanBase<T, Layout<S::Dim, F, O>> {
    type Output = T;

    fn index(&self, index: S) -> &T {
        let layout = self.layout();

        for i in 0..S::Dim::RANK {
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

        for i in 0..S::Dim::RANK {
            if index[i] >= layout.size(i) {
                panic_bounds_check(index[i], layout.size(i))
            }
        }

        unsafe { &mut *self.as_mut_ptr().offset(layout.offset(index)) }
    }
}

impl<T, D: Dim, F: Uniform, O: Order> Index<usize> for SpanBase<T, Layout<D, F, O>> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &*self.as_ptr().offset(self.stride(D::dim::<O>(0)) * index as isize) }
    }
}

impl<T, D: Dim, F: Uniform, O: Order> IndexMut<usize> for SpanBase<T, Layout<D, F, O>> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &mut *self.as_mut_ptr().offset(self.stride(D::dim::<O>(0)) * index as isize) }
    }
}

macro_rules! impl_index_range {
    ($type:ty) => {
        impl<T, D: Dim, O: Order> Index<$type> for DenseSpan<T, D, O> {
            type Output = [T];

            fn index(&self, index: $type) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl<T, D: Dim, O: Order> IndexMut<$type> for DenseSpan<T, D, O> {
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
