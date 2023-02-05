use std::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::array::SpanArray;
use crate::dim::{Dim, Shape};
use crate::format::{Dense, Format, Uniform};
use crate::layout::panic_bounds_check;

/// Array span index trait, for an element or a subslice.
pub trait SpanIndex<T, D: Dim, F: Format> {
    /// Array element or subslice type.
    type Output: ?Sized;

    #[doc(hidden)]
    unsafe fn get_unchecked(self, span: &SpanArray<T, D, F>) -> &Self::Output;

    #[doc(hidden)]
    unsafe fn get_unchecked_mut(self, span: &mut SpanArray<T, D, F>) -> &mut Self::Output;

    #[doc(hidden)]
    fn index(self, span: &SpanArray<T, D, F>) -> &Self::Output;

    #[doc(hidden)]
    fn index_mut(self, span: &mut SpanArray<T, D, F>) -> &mut Self::Output;
}

impl<T, S: Shape, D: Dim<Shape = S>, F: Format> SpanIndex<T, D, F> for S {
    type Output = T;

    unsafe fn get_unchecked(self, span: &SpanArray<T, D, F>) -> &T {
        &*span.as_ptr().offset(span.layout().offset(self))
    }

    unsafe fn get_unchecked_mut(self, span: &mut SpanArray<T, D, F>) -> &mut T {
        &mut *span.as_mut_ptr().offset(span.layout().offset(self))
    }

    fn index(self, span: &SpanArray<T, D, F>) -> &T {
        for i in 0..D::RANK {
            if self[i] >= span.size(i) {
                panic_bounds_check(self[i], span.size(i))
            }
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut SpanArray<T, D, F>) -> &mut T {
        for i in 0..D::RANK {
            if self[i] >= span.size(i) {
                panic_bounds_check(self[i], span.size(i))
            }
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

impl<T, D: Dim, F: Uniform> SpanIndex<T, D, F> for usize {
    type Output = T;

    unsafe fn get_unchecked(self, span: &SpanArray<T, D, F>) -> &T {
        debug_assert!(self < span.len(), "index out of bounds");

        let offset = if D::RANK > 0 { span.stride(0) * self as isize } else { 0 };

        &*span.as_ptr().offset(offset)
    }

    unsafe fn get_unchecked_mut(self, span: &mut SpanArray<T, D, F>) -> &mut T {
        debug_assert!(self < span.len(), "index out of bounds");

        let offset = if D::RANK > 0 { span.stride(0) * self as isize } else { 0 };

        &mut *span.as_mut_ptr().offset(offset)
    }

    fn index(self, span: &SpanArray<T, D, F>) -> &T {
        if self >= span.len() {
            panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut SpanArray<T, D, F>) -> &mut T {
        if self >= span.len() {
            panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

macro_rules! impl_span_index {
    ($type:ty) => {
        impl<T, D: Dim> SpanIndex<T, D, Dense> for $type {
            type Output = [T];

            unsafe fn get_unchecked(self, span: &SpanArray<T, D, Dense>) -> &[T] {
                span.as_slice().get_unchecked(self)
            }

            unsafe fn get_unchecked_mut(self, span: &mut SpanArray<T, D, Dense>) -> &mut [T] {
                span.as_mut_slice().get_unchecked_mut(self)
            }

            fn index(self, span: &SpanArray<T, D, Dense>) -> &[T] {
                span.as_slice().index(self)
            }

            fn index_mut(self, span: &mut SpanArray<T, D, Dense>) -> &mut [T] {
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
