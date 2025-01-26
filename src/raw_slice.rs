use std::mem;
use std::ptr::NonNull;

use crate::layout::Layout;
use crate::shape::Shape;
use crate::slice::Slice;

pub(crate) struct RawSlice<T, S: Shape, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<S>,
}

impl<T, S: Shape, L: Layout> RawSlice<T, S, L> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut Slice<T, S, L> {
        if mem::size_of::<L::Mapping<S>>() > 0 {
            unsafe { &mut *(self as *mut Self as *mut Slice<T, S, L>) }
        } else {
            unsafe { &mut *(self.ptr.as_ptr() as *mut Slice<T, S, L>) }
        }
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_slice(&self) -> &Slice<T, S, L> {
        if mem::size_of::<L::Mapping<S>>() > 0 {
            unsafe { &*(self as *const Self as *const Slice<T, S, L>) }
        } else {
            unsafe { &*(self.ptr.as_ptr() as *const Slice<T, S, L>) }
        }
    }

    pub(crate) fn from_mut_slice(slice: &mut Slice<T, S, L>) -> &mut Self {
        assert!(mem::size_of::<L::Mapping<S>>() > 0, "ZST not allowed");

        unsafe { &mut *(slice as *mut Slice<T, S, L> as *mut Self) }
    }

    pub(crate) fn from_slice(slice: &Slice<T, S, L>) -> &Self {
        assert!(mem::size_of::<L::Mapping<S>>() > 0, "ZST not allowed");

        unsafe { &*(slice as *const Slice<T, S, L> as *const Self) }
    }

    pub(crate) fn mapping(&self) -> &L::Mapping<S> {
        &self.mapping
    }

    pub(crate) unsafe fn mapping_mut(&mut self) -> &mut L::Mapping<S> {
        &mut self.mapping
    }

    pub(crate) unsafe fn new_unchecked(ptr: *mut T, mapping: L::Mapping<S>) -> Self {
        unsafe { Self { ptr: NonNull::new_unchecked(ptr), mapping } }
    }

    pub(crate) unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
        self.ptr = unsafe { NonNull::new_unchecked(new_ptr) };
    }
}

impl<T, S: Shape, L: Layout> Clone for RawSlice<T, S, L> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, mapping: self.mapping.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.ptr = source.ptr;
        self.mapping.clone_from(&source.mapping);
    }
}

impl<T, S: Shape, L: Layout<Mapping<S>: Copy>> Copy for RawSlice<T, S, L> {}
