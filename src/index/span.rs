use std::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::Dims;
use crate::index;
use crate::layout::{Dense, Layout, Uniform};
use crate::mapping::Mapping;
use crate::shape::Shape;
use crate::span::Span;

/// Array span index trait, for an element or a subslice.
pub trait SpanIndex<T, S: Shape, L: Layout> {
    /// Array element or subslice type.
    type Output: ?Sized;

    #[doc(hidden)]
    unsafe fn get_unchecked(self, span: &Span<T, S, L>) -> &Self::Output;

    #[doc(hidden)]
    unsafe fn get_unchecked_mut(self, span: &mut Span<T, S, L>) -> &mut Self::Output;

    #[doc(hidden)]
    fn index(self, span: &Span<T, S, L>) -> &Self::Output;

    #[doc(hidden)]
    fn index_mut(self, span: &mut Span<T, S, L>) -> &mut Self::Output;
}

impl<T, D: Dims, S: Shape<Dims = D>, L: Layout> SpanIndex<T, S, L> for D {
    type Output = T;

    unsafe fn get_unchecked(self, span: &Span<T, S, L>) -> &T {
        &*span.as_ptr().offset(span.mapping().offset(self))
    }

    unsafe fn get_unchecked_mut(self, span: &mut Span<T, S, L>) -> &mut T {
        &mut *span.as_mut_ptr().offset(span.mapping().offset(self))
    }

    fn index(self, span: &Span<T, S, L>) -> &T {
        let dims = span.dims();

        for i in 0..S::RANK {
            if self[i] >= dims[i] {
                index::panic_bounds_check(self[i], dims[i])
            }
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut Span<T, S, L>) -> &mut T {
        let dims = span.dims();

        for i in 0..S::RANK {
            if self[i] >= dims[i] {
                index::panic_bounds_check(self[i], dims[i])
            }
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

impl<T, S: Shape, L: Uniform> SpanIndex<T, S, L> for usize {
    type Output = T;

    unsafe fn get_unchecked(self, span: &Span<T, S, L>) -> &T {
        debug_assert!(self < span.len(), "index out of bounds");

        let offset = if S::RANK > 0 { span.stride(0) * self as isize } else { 0 };

        &*span.as_ptr().offset(offset)
    }

    unsafe fn get_unchecked_mut(self, span: &mut Span<T, S, L>) -> &mut T {
        debug_assert!(self < span.len(), "index out of bounds");

        let offset = if S::RANK > 0 { span.stride(0) * self as isize } else { 0 };

        &mut *span.as_mut_ptr().offset(offset)
    }

    fn index(self, span: &Span<T, S, L>) -> &T {
        if self >= span.len() {
            index::panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked(span) }
    }

    fn index_mut(self, span: &mut Span<T, S, L>) -> &mut T {
        if self >= span.len() {
            index::panic_bounds_check(self, span.len())
        }

        unsafe { self.get_unchecked_mut(span) }
    }
}

macro_rules! impl_span_index {
    ($type:ty) => {
        impl<T, S: Shape> SpanIndex<T, S, Dense> for $type {
            type Output = [T];

            unsafe fn get_unchecked(self, span: &Span<T, S>) -> &[T] {
                span.as_slice().get_unchecked(self)
            }

            unsafe fn get_unchecked_mut(self, span: &mut Span<T, S>) -> &mut [T] {
                span.as_mut_slice().get_unchecked_mut(self)
            }

            fn index(self, span: &Span<T, S>) -> &[T] {
                span.as_slice().index(self)
            }

            fn index_mut(self, span: &mut Span<T, S>) -> &mut [T] {
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
