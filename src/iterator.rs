use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::grid::{SubGrid, SubGridMut};
use crate::layout::Layout;

pub struct AxisIter<'a, T, L: Layout> {
    ptr: NonNull<T>,
    layout: L,
    index: usize,
    size: usize,
    stride: isize,
    _marker: PhantomData<&'a T>,
}

pub struct AxisIterMut<'a, T, L: Layout> {
    ptr: NonNull<T>,
    layout: L,
    index: usize,
    size: usize,
    stride: isize,
    _marker: PhantomData<&'a mut T>,
}

pub struct StridedIter<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    _marker: PhantomData<&'a T>,
}

pub struct StridedIterMut<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    _marker: PhantomData<&'a mut T>,
}

macro_rules! impl_axis_iter {
    ($type:ty, $grid:tt, $raw_mut:tt) => {
        impl<'a, T, L: Layout> $type {
            pub unsafe fn new(
                ptr: *$raw_mut T,
                layout: L,
                size: usize,
                stride: isize,
            ) -> Self {
                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    layout,
                    index: 0,
                    size,
                    stride,
                    _marker: PhantomData,
                }
            }
        }

        impl<'a, T, L: Layout> DoubleEndedIterator for $type {
            #[inline(always)]
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    self.size -= 1;

                    let count = self.stride * self.size as isize;

                    unsafe { Some($grid::new(self.ptr.as_ptr().offset(count), self.layout)) }
                }
            }
        }

        impl<'a, T, L: Layout> ExactSizeIterator for $type {}
        impl<'a, T, L: Layout> FusedIterator for $type {}

        impl<'a, T, L: Layout> Iterator for $type {
            type Item = $grid<'a, T, L>;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    let count = self.stride * self.index as isize;

                    self.index += 1;

                    unsafe { Some($grid::new(self.ptr.as_ptr().offset(count), self.layout)) }
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.size - self.index;

                (len, Some(len))
            }
       }
    }
}

impl_axis_iter!(AxisIter<'a, T, L>, SubGrid, const);
impl_axis_iter!(AxisIterMut<'a, T, L>, SubGridMut, mut);

impl<'a, T, L: Layout> Clone for AxisIter<'a, T, L> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            layout: self.layout,
            index: self.index,
            size: self.size,
            stride: self.stride,
            _marker: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync, L: Layout> Send for AxisIter<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Layout> Sync for AxisIter<'a, T, L> {}

unsafe impl<'a, T: Send, L: Layout> Send for AxisIterMut<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Layout> Sync for AxisIterMut<'a, T, L> {}

macro_rules! impl_strided_iter {
    ($type:ty, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T> $type {
            pub unsafe fn new(
                ptr: *$raw_mut T,
                size: usize,
                stride: isize,
            ) -> Self {
                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    index: 0,
                    size,
                    stride,
                    _marker: PhantomData,
                }
            }
        }

        impl<'a, T> DoubleEndedIterator for $type {
            #[inline(always)]
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    self.size -= 1;

                    let count = self.stride * self.size as isize;

                    unsafe { Some(&$($mut)? *self.ptr.as_ptr().offset(count)) }
                }
            }
        }

        impl<'a, T> ExactSizeIterator for $type {}
        impl<'a, T> FusedIterator for $type {}

        impl<'a, T> Iterator for $type {
            type Item = &'a $($mut)? T;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    let count = self.stride * self.index as isize;

                    self.index += 1;

                    unsafe { Some(&$($mut)? *self.ptr.as_ptr().offset(count)) }
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.size - self.index;

                (len, Some(len))
            }
       }
    }
}

impl_strided_iter!(StridedIter<'a, T>, const, {});
impl_strided_iter!(StridedIterMut<'a, T>, mut, {mut});

impl<'a, T> Clone for StridedIter<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            index: self.index,
            size: self.size,
            stride: self.stride,
            _marker: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync> Send for StridedIter<'a, T> {}
unsafe impl<'a, T: Sync> Sync for StridedIter<'a, T> {}

unsafe impl<'a, T: Send> Send for StridedIterMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for StridedIterMut<'a, T> {}
