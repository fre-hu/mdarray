use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::grid::{SubGrid, SubGridMut};

pub struct AxisIter<'a, T, L: Copy> {
    ptr: NonNull<T>,
    layout: L,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

pub struct AxisIterMut<'a, T, L: Copy> {
    ptr: NonNull<T>,
    layout: L,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

pub struct LinearIter<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

pub struct LinearIterMut<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

macro_rules! impl_axis_iter {
    ($type:ty, $grid:tt, $raw_mut:tt) => {
        impl<'a, T, L: Copy> $type {
            pub unsafe fn new_unchecked(
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
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, T, L: Copy> DoubleEndedIterator for $type {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    self.size -= 1;

                    let count = self.stride * self.size as isize;

                    unsafe {
                        Some($grid::new_unchecked(self.ptr.as_ptr().offset(count), self.layout))
                    }
                }
            }
        }

        impl<'a, T, L: Copy> ExactSizeIterator for $type {}
        impl<'a, T, L: Copy> FusedIterator for $type {}

        impl<'a, T, L: Copy> Iterator for $type {
            type Item = $grid<'a, T, L>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    let count = self.stride * self.index as isize;

                    self.index += 1;

                    unsafe {
                        Some($grid::new_unchecked(self.ptr.as_ptr().offset(count), self.layout))
                    }
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

impl<'a, T, L: Copy> Clone for AxisIter<'a, T, L> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            layout: self.layout,
            index: self.index,
            size: self.size,
            stride: self.stride,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync, L: Copy> Send for AxisIter<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Copy> Sync for AxisIter<'a, T, L> {}

unsafe impl<'a, T: Send, L: Copy> Send for AxisIterMut<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Copy> Sync for AxisIterMut<'a, T, L> {}

macro_rules! impl_linear_iter {
    ($type:ty, $raw_mut:tt, {$($const:tt)?}, {$($mut:tt)?}) => {
        impl<'a, T> $type {
            pub $($const)? unsafe fn new_unchecked(
                ptr: *$raw_mut T,
                size: usize,
                stride: isize,
            ) -> Self {
                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    index: 0,
                    size,
                    stride,
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, T> DoubleEndedIterator for $type {
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

impl_linear_iter!(LinearIter<'a, T>, const, {const}, {});
impl_linear_iter!(LinearIterMut<'a, T>, mut, {}, {mut});

impl<'a, T> Clone for LinearIter<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            index: self.index,
            size: self.size,
            stride: self.stride,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync> Send for LinearIter<'a, T> {}
unsafe impl<'a, T: Sync> Sync for LinearIter<'a, T> {}

unsafe impl<'a, T: Send> Send for LinearIterMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for LinearIterMut<'a, T> {}
