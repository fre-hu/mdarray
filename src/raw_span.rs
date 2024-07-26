use std::mem;
use std::ptr::NonNull;

use crate::dim::Dim;
use crate::layout::Layout;
use crate::span::Span;

pub(crate) struct RawSpan<T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D>,
}

impl<T, D: Dim, L: Layout> RawSpan<T, D, L> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_span(&mut self) -> &mut Span<T, D, L> {
        if D::RANK > 0 {
            unsafe { &mut *(self as *mut Self as *mut Span<T, D, L>) }
        } else {
            unsafe { &mut *(self.ptr.as_ptr() as *mut Span<T, D, L>) }
        }
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_span(&self) -> &Span<T, D, L> {
        if D::RANK > 0 {
            unsafe { &*(self as *const Self as *const Span<T, D, L>) }
        } else {
            unsafe { &*(self.ptr.as_ptr() as *const Span<T, D, L>) }
        }
    }

    pub(crate) fn from_mut_span(span: &mut Span<T, D, L>) -> &mut Self {
        assert!(D::RANK > 0, "invalid rank");

        unsafe { &mut *(span as *mut Span<T, D, L> as *mut Self) }
    }

    pub(crate) fn from_span(span: &Span<T, D, L>) -> &Self {
        assert!(D::RANK > 0, "invalid rank");

        unsafe { &*(span as *const Span<T, D, L> as *const Self) }
    }

    pub(crate) fn mapping(&self) -> L::Mapping<D> {
        self.mapping
    }

    pub(crate) unsafe fn new_unchecked(ptr: *mut T, mapping: L::Mapping<D>) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { ptr: NonNull::new_unchecked(ptr), mapping }
    }

    pub(crate) unsafe fn set_mapping(&mut self, new_mapping: L::Mapping<D>) {
        self.mapping = new_mapping;
    }

    pub(crate) unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
        self.ptr = NonNull::new_unchecked(new_ptr);
    }
}

impl<T, D: Dim, L: Layout> Clone for RawSpan<T, D, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, D: Dim, L: Layout> Copy for RawSpan<T, D, L> {}
