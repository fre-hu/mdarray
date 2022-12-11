use std::mem;
use std::ptr::NonNull;

use crate::dim::Dim;
use crate::format::Format;
use crate::layout::Layout;
use crate::span::SpanBase;

pub struct RawSpan<T, D: Dim, F: Format> {
    ptr: NonNull<T>,
    layout: Layout<D, F>,
}

impl<T, D: Dim, F: Format> RawSpan<T, D, F> {
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_span(&mut self) -> &mut SpanBase<T, D, F> {
        #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
        let _ = self as *mut Self as usize; // Expose pointer provenance, see UCG issue #256.

        unsafe { &mut *(self as *mut Self as *mut SpanBase<T, D, F>) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_span(&self) -> &SpanBase<T, D, F> {
        #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
        let _ = self as *const Self as usize; // Expose pointer provenance, see UCG issue #256.

        unsafe { &*(self as *const Self as *const SpanBase<T, D, F>) }
    }

    pub fn layout(&self) -> Layout<D, F> {
        self.layout
    }

    #[cfg(any(feature = "nightly", not(feature = "permissive-provenance")))]
    pub fn from_mut_span(span: &mut SpanBase<T, D, F>) -> &mut Self {
        unsafe { &mut *(span as *mut SpanBase<T, D, F> as *mut Self) } // Use existing provenance.
    }

    #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
    pub fn from_mut_span(span: &mut SpanBase<T, D, F>) -> &mut Self {
        unsafe {
            let ptr = span as *mut SpanBase<T, D, F>;

            &mut *(ptr as usize as *mut Self) // Use exposed provenance, see UCG issue #256.
        }
    }

    #[cfg(any(feature = "nightly", not(feature = "permissive-provenance")))]
    pub fn from_span(span: &SpanBase<T, D, F>) -> &Self {
        unsafe { &*(span as *const SpanBase<T, D, F> as *const Self) } // Use existing provenance.
    }

    #[cfg(all(not(feature = "nightly"), feature = "permissive-provenance"))]
    pub fn from_span(span: &SpanBase<T, D, F>) -> &Self {
        unsafe {
            let ptr = span as *const SpanBase<T, D, F>;

            &*(ptr as usize as *const Self) // Use exposed provenance, see UCG issue #256.
        }
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
