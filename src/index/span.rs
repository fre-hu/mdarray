use std::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::{Dim, Shape};
use crate::format::{Format, Uniform};
use crate::layout::{panic_bounds_check, DenseLayout, Layout};
use crate::span::{DenseSpan, SpanBase};

/// Array span index trait, for an element or a subslice.
pub trait SpanIndex<T, L: Copy> {
    /// Array element or subslice type.
    type Output: ?Sized;

    #[doc(hidden)]
    unsafe fn get_unchecked(self, span: &SpanBase<T, L>) -> &Self::Output;

    #[doc(hidden)]
    unsafe fn get_unchecked_mut(self, span: &mut SpanBase<T, L>) -> &mut Self::Output;

    #[doc(hidden)]
    fn index(self, span: &SpanBase<T, L>) -> &Self::Output;

    #[doc(hidden)]
    fn index_mut(self, span: &mut SpanBase<T, L>) -> &mut Self::Output;
}

impl<T, S: Shape, D: Dim<Shape = S>, F: Format> SpanIndex<T, Layout<D, F>> for S {
    type Output = T;

    unsafe fn get_unchecked(self, span: &SpanBase<T, Layout<D, F>>) -> &T {
        &*span.as_ptr().offset(span.layout().offset(self))
    }

    unsafe fn get_unchecked_mut(self, span: &mut SpanBase<T, Layout<D, F>>) -> &mut T {
        &mut *span.as_mut_ptr().offset(span.layout().offset(self))
    }

    fn index(self, span: &SpanBase<T, Layout<D, F>>) -> &T {
        for i in 0..D::RANK {
            if self[i] >= span.size(i) {
                panic_bounds_check(self[i], span.size(i))
            }
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut SpanBase<T, Layout<D, F>>) -> &mut T {
        for i in 0..D::RANK {
            if self[i] >= span.size(i) {
                panic_bounds_check(self[i], span.size(i))
            }
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

impl<T, D: Dim, F: Uniform> SpanIndex<T, Layout<D, F>> for usize {
    type Output = T;

    unsafe fn get_unchecked(self, span: &SpanBase<T, Layout<D, F>>) -> &T {
        debug_assert!(self < span.len(), "index out of bounds");

        &*span.as_ptr().offset(span.stride(D::dim(0)) * self as isize)
    }

    unsafe fn get_unchecked_mut(self, span: &mut SpanBase<T, Layout<D, F>>) -> &mut T {
        debug_assert!(self < span.len(), "index out of bounds");

        &mut *span.as_mut_ptr().offset(span.stride(D::dim(0)) * self as isize)
    }

    fn index(self, span: &SpanBase<T, Layout<D, F>>) -> &T {
        if self >= span.len() {
            panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut SpanBase<T, Layout<D, F>>) -> &mut T {
        if self >= span.len() {
            panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

macro_rules! impl_span_index {
    ($type:ty) => {
        impl<T, D: Dim> SpanIndex<T, DenseLayout<D>> for $type {
            type Output = [T];

            unsafe fn get_unchecked(self, span: &DenseSpan<T, D>) -> &[T] {
                span.as_slice().get_unchecked(self)
            }

            unsafe fn get_unchecked_mut(self, span: &mut DenseSpan<T, D>) -> &mut [T] {
                span.as_mut_slice().get_unchecked_mut(self)
            }

            fn index(self, span: &DenseSpan<T, D>) -> &[T] {
                span.as_slice().index(self)
            }

            fn index_mut(self, span: &mut DenseSpan<T, D>) -> &mut [T] {
                span.as_mut_slice().index_mut(self)
            }
        }
    };
}

impl_span_index!((Bound<usize>, Bound<usize>));
impl_span_index!(Range<usize>);
impl_span_index!(RangeFrom<usize>);
impl_span_index!(RangeFull);
impl_span_index!(RangeInclusive<usize>);
impl_span_index!(RangeTo<usize>);
impl_span_index!(RangeToInclusive<usize>);
