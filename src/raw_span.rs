use std::mem;
use std::ptr::NonNull;

use crate::layout::Layout;
use crate::shape::Shape;
use crate::span::Span;

pub(crate) struct RawSpan<T, S: Shape, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<S>,
}

impl<T, S: Shape, L: Layout> RawSpan<T, S, L> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_span(&mut self) -> &mut Span<T, S, L> {
        if mem::size_of::<S>() > 0 {
            unsafe { &mut *(self as *mut Self as *mut Span<T, S, L>) }
        } else {
            unsafe { &mut *(self.ptr.as_ptr() as *mut Span<T, S, L>) }
        }
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_span(&self) -> &Span<T, S, L> {
        if mem::size_of::<S>() > 0 {
            unsafe { &*(self as *const Self as *const Span<T, S, L>) }
        } else {
            unsafe { &*(self.ptr.as_ptr() as *const Span<T, S, L>) }
        }
    }

    pub(crate) fn from_mut_span(span: &mut Span<T, S, L>) -> &mut Self {
        assert!(mem::size_of::<S>() > 0, "ZST not allowed");

        unsafe { &mut *(span as *mut Span<T, S, L> as *mut Self) }
    }

    pub(crate) fn from_span(span: &Span<T, S, L>) -> &Self {
        assert!(mem::size_of::<S>() > 0, "ZST not allowed");

        unsafe { &*(span as *const Span<T, S, L> as *const Self) }
    }

    pub(crate) fn mapping(&self) -> L::Mapping<S> {
        self.mapping
    }

    pub(crate) unsafe fn new_unchecked(ptr: *mut T, mapping: L::Mapping<S>) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { ptr: NonNull::new_unchecked(ptr), mapping }
    }

    pub(crate) unsafe fn set_mapping(&mut self, new_mapping: L::Mapping<S>) {
        self.mapping = new_mapping;
    }

    pub(crate) unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
        self.ptr = NonNull::new_unchecked(new_ptr);
    }
}

impl<T, S: Shape, L: Layout> Clone for RawSpan<T, S, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, S: Shape, L: Layout> Copy for RawSpan<T, S, L> {}
