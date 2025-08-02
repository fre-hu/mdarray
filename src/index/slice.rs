use core::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::index;
use crate::layout::{Dense, Layout};
use crate::mapping::Mapping;
use crate::shape::Shape;
use crate::slice::Slice;

/// Array slice index trait, for an element or a subslice.
pub trait SliceIndex<T, S: Shape, L: Layout> {
    /// Array element or subslice type.
    type Output: ?Sized;

    #[doc(hidden)]
    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &Self::Output;

    #[doc(hidden)]
    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut Self::Output;

    #[doc(hidden)]
    fn index(self, slice: &Slice<T, S, L>) -> &Self::Output;

    #[doc(hidden)]
    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut Self::Output;
}

impl<T, S: Shape, L: Layout> SliceIndex<T, S, L> for &[usize] {
    type Output = T;

    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &T {
        unsafe { &*slice.as_ptr().offset(slice.mapping().offset(self)) }
    }

    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        unsafe { &mut *slice.as_mut_ptr().offset(slice.mapping().offset(self)) }
    }

    fn index(self, slice: &Slice<T, S, L>) -> &T {
        assert!(self.len() == slice.rank(), "invalid rank");

        for i in 0..self.len() {
            if self[i] >= slice.dim(i) {
                index::panic_bounds_check(self[i], slice.dim(i));
            }
        }

        unsafe { SliceIndex::get_unchecked(self, slice) }
    }

    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        assert!(self.len() == slice.rank(), "invalid rank");

        for i in 0..self.len() {
            if self[i] >= slice.dim(i) {
                index::panic_bounds_check(self[i], slice.dim(i));
            }
        }

        unsafe { SliceIndex::get_unchecked_mut(self, slice) }
    }
}

impl<T, const N: usize, S: Shape, L: Layout> SliceIndex<T, S, L> for [usize; N] {
    type Output = T;

    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &T {
        unsafe { SliceIndex::get_unchecked(&self[..], slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        unsafe { SliceIndex::get_unchecked_mut(&self[..], slice) }
    }

    fn index(self, slice: &Slice<T, S, L>) -> &T {
        SliceIndex::index(&self[..], slice)
    }

    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        SliceIndex::index_mut(&self[..], slice)
    }
}

impl<T, S: Shape, L: Layout> SliceIndex<T, S, L> for usize {
    type Output = T;

    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &T {
        unsafe { &*slice.as_ptr().offset(slice.mapping().linear_offset(self)) }
    }

    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        unsafe { &mut *slice.as_mut_ptr().offset(slice.mapping().linear_offset(self)) }
    }

    fn index(self, slice: &Slice<T, S, L>) -> &T {
        if self >= slice.len() {
            index::panic_bounds_check(self, slice.len());
        }

        unsafe { SliceIndex::get_unchecked(self, slice) }
    }

    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        if self >= slice.len() {
            index::panic_bounds_check(self, slice.len());
        }

        unsafe { SliceIndex::get_unchecked_mut(self, slice) }
    }
}

macro_rules! impl_slice_index {
    ($type:ty) => {
        impl<T, S: Shape> SliceIndex<T, S, Dense> for $type {
            type Output = [T];

            unsafe fn get_unchecked(self, slice: &Slice<T, S>) -> &[T] {
                unsafe { <&[T]>::from(slice.flatten()).get_unchecked(self) }
            }

            unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S>) -> &mut [T] {
                unsafe { <&mut [T]>::from(slice.flatten_mut()).get_unchecked_mut(self) }
            }

            fn index(self, slice: &Slice<T, S>) -> &[T] {
                <&[T]>::from(slice.flatten()).index(self)
            }

            fn index_mut(self, slice: &mut Slice<T, S>) -> &mut [T] {
                <&mut [T]>::from(slice.flatten_mut()).index_mut(self)
            }
        }
    };
}

impl_slice_index!((Bound<usize>, Bound<usize>));
impl_slice_index!(Range<usize>);
impl_slice_index!(RangeFrom<usize>);
impl_slice_index!(RangeFull);
impl_slice_index!(RangeInclusive<usize>);
impl_slice_index!(RangeTo<usize>);
impl_slice_index!(RangeToInclusive<usize>);
