use std::mem;
use std::ptr::NonNull;

use crate::array::SpanArray;
use crate::dim::Dim;
use crate::format::Format;
use crate::layout::Layout;

pub struct RawSpan<T, D: Dim, F: Format> {
    ptr: NonNull<T>,
    layout: Layout<D, F>,
}

impl<T, D: Dim, F: Format> RawSpan<T, D, F> {
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_span(&mut self) -> &mut SpanArray<T, D, F> {
        if D::RANK > 0 {
            unsafe { &mut *(self as *mut Self as *mut SpanArray<T, D, F>) }
        } else {
            unsafe { &mut *(self.ptr.as_ptr() as *mut SpanArray<T, D, F>) }
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_span(&self) -> &SpanArray<T, D, F> {
        if D::RANK > 0 {
            unsafe { &*(self as *const Self as *const SpanArray<T, D, F>) }
        } else {
            unsafe { &*(self.ptr.as_ptr() as *const SpanArray<T, D, F>) }
        }
    }

    pub fn layout(&self) -> Layout<D, F> {
        self.layout
    }

    pub fn from_mut_span(span: &mut SpanArray<T, D, F>) -> &mut Self {
        assert!(D::RANK > 0, "invalid rank");

        unsafe { &mut *(span as *mut SpanArray<T, D, F> as *mut Self) }
    }

    pub fn from_span(span: &SpanArray<T, D, F>) -> &Self {
        assert!(D::RANK > 0, "invalid rank");

        unsafe { &*(span as *const SpanArray<T, D, F> as *const Self) }
    }

    pub unsafe fn new_unchecked(ptr: *mut T, layout: Layout<D, F>) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { ptr: NonNull::new_unchecked(ptr), layout }
    }

    pub unsafe fn set_layout(&mut self, new_layout: Layout<D, F>) {
        self.layout = new_layout;
    }

    pub unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
        self.ptr = NonNull::new_unchecked(new_ptr);
    }
}

impl<T, D: Dim, F: Format> Clone for RawSpan<T, D, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, D: Dim, F: Format> Copy for RawSpan<T, D, F> {}
