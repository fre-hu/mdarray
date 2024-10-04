use std::ops::{
    Bound, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::dim::Dims;
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

impl<T, D: Dims, S: Shape<Dims = D>, L: Layout> SliceIndex<T, S, L> for D {
    type Output = T;

    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &T {
        &*slice.as_ptr().offset(slice.mapping().offset(self))
    }

    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        &mut *slice.as_mut_ptr().offset(slice.mapping().offset(self))
    }

    fn index(self, slice: &Slice<T, S, L>) -> &T {
        let dims = slice.dims();

        for i in 0..S::RANK {
            if self[i] >= dims[i] {
                index::panic_bounds_check(self[i], dims[i])
            }
        }

        unsafe { self.get_unchecked(slice) }
    }

    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        let dims = slice.dims();

        for i in 0..S::RANK {
            if self[i] >= dims[i] {
                index::panic_bounds_check(self[i], dims[i])
            }
        }

        unsafe { self.get_unchecked_mut(slice) }
    }
}

impl<T, S: Shape, L: Layout> SliceIndex<T, S, L> for usize {
    type Output = T;

    unsafe fn get_unchecked(self, slice: &Slice<T, S, L>) -> &T {
        &*slice.as_ptr().offset(slice.mapping().linear_offset(self))
    }

    unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        &mut *slice.as_mut_ptr().offset(slice.mapping().linear_offset(self))
    }

    fn index(self, slice: &Slice<T, S, L>) -> &T {
        if self >= slice.len() {
            index::panic_bounds_check(self, slice.len())
        }

        unsafe { self.get_unchecked(slice) }
    }

    fn index_mut(self, slice: &mut Slice<T, S, L>) -> &mut T {
        if self >= slice.len() {
            index::panic_bounds_check(self, slice.len())
        }

        unsafe { self.get_unchecked_mut(slice) }
    }
}

macro_rules! impl_slice_index {
    ($type:ty) => {
        impl<T, S: Shape> SliceIndex<T, S, Dense> for $type {
            type Output = [T];

            unsafe fn get_unchecked(self, slice: &Slice<T, S>) -> &[T] {
                AsRef::<[T]>::as_ref(slice).get_unchecked(self)
            }

            unsafe fn get_unchecked_mut(self, slice: &mut Slice<T, S>) -> &mut [T] {
                AsMut::<[T]>::as_mut(slice).get_unchecked_mut(self)
            }

            fn index(self, slice: &Slice<T, S>) -> &[T] {
                AsRef::<[T]>::as_ref(slice).index(self)
            }

            fn index_mut(self, slice: &mut Slice<T, S>) -> &mut [T] {
                AsMut::<[T]>::as_mut(slice).index_mut(self)
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
