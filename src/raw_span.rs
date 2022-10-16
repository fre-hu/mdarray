use std::mem;
use std::ptr::NonNull;

use crate::span::SpanBase;

pub struct RawSpan<T, L: Copy> {
    ptr: NonNull<T>,
    layout: L,
}

impl<T, L: Copy> RawSpan<T, L> {
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_span(&mut self) -> &mut SpanBase<T, L> {
        #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
        let _ = self as *mut Self as usize; // Expose pointer provenance, see UCG issue #256.

        unsafe { &mut *(self as *mut Self as *mut SpanBase<T, L>) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_span(&self) -> &SpanBase<T, L> {
        #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
        let _ = self as *const Self as usize; // Expose pointer provenance, see UCG issue #256.

        unsafe { &*(self as *const Self as *const SpanBase<T, L>) }
    }

    pub fn layout(&self) -> L {
        self.layout
    }

    #[cfg(any(feature = "nightly", not(feature = "permissive-provenance")))]
    pub fn from_mut_span(span: &mut SpanBase<T, L>) -> &mut Self {
        unsafe { &mut *(span as *mut SpanBase<T, L> as *mut Self) } // Use existing provenance.
    }

    #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
    pub fn from_mut_span(span: &mut SpanBase<T, L>) -> &mut Self {
        unsafe {
            let ptr = span as *mut SpanBase<T, L>;

            &mut *(ptr as usize as *mut Self) // Use exposed provenance, see UCG issue #256.
        }
    }

    #[cfg(any(feature = "nightly", not(feature = "permissive-provenance")))]
    pub fn from_span(span: &SpanBase<T, L>) -> &Self {
        unsafe { &*(span as *const SpanBase<T, L> as *const Self) } // Use existing provenance.
    }

    #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
    pub fn from_span(span: &SpanBase<T, L>) -> &Self {
        unsafe {
            let ptr = span as *const SpanBase<T, L>;

            &*(ptr as usize as *const Self) // Use exposed provenance, see UCG issue #256.
        }
    }

    pub unsafe fn new_unchecked(ptr: *mut T, layout: L) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { ptr: NonNull::new_unchecked(ptr), layout }
    }

    pub unsafe fn set_layout(&mut self, new_layout: L) {
        self.layout = new_layout;
    }

    pub unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
        self.ptr = NonNull::new_unchecked(new_ptr);
    }
}

impl<T, L: Copy> Clone for RawSpan<T, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, L: Copy> Copy for RawSpan<T, L> {}
