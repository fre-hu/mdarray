use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::array::{ViewArray, ViewArrayMut};
use crate::dim::Dim;
use crate::layout::Layout;

/// Array axis iterator.
pub struct AxisIter<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

/// Mutable array axis iterator.
pub struct AxisIterMut<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

/// Flat array span iterator.
pub struct FlatIter<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

/// Mutable flat array span iterator.
pub struct FlatIterMut<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

macro_rules! impl_axis_iter {
    ($name:tt, $grid:tt, $raw_mut:tt) => {
        impl<'a, T, D: Dim, L: Layout> $name<'a, T, D, L> {
            pub(crate) unsafe fn new_unchecked(
                ptr: *$raw_mut T,
                mapping: L::Mapping<D>,
                size: usize,
                stride: isize,
            ) -> Self {
                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    mapping,
                    index: 0,
                    size,
                    stride,
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, T: Debug, D: Dim, L: Layout> Debug for $name<'a, T, D, L> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                struct Value<'b, 'c, U, E: Dim, M: Layout>(&'b $name<'c, U, E, M>);

                impl<'b, 'c, U: Debug, E: Dim, M: Layout> Debug for Value<'b, 'c, U, E, M> {
                    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                        let mut list = f.debug_list();

                        for i in self.0.index..self.0.size {
                            unsafe {
                                let ptr = self.0.ptr.as_ptr().offset(self.0.stride * i as isize);
                                let view = ViewArray::<U, E, M>::new_unchecked(ptr, self.0.mapping);
                                let _ = list.entry(&view);
                            }
                        }

                        list.finish()
                    }
                }

                f.debug_tuple(stringify!($name)).field(&Value(self)).finish()
            }
        }

        impl<'a, T, D: Dim, L: Layout> DoubleEndedIterator for $name<'a, T, D, L> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    self.size -= 1;

                    let count = self.stride * self.size as isize;

                    unsafe {
                        Some($grid::new_unchecked(self.ptr.as_ptr().offset(count), self.mapping))
                    }
                }
            }
        }

        impl<'a, T, D: Dim, L: Layout> ExactSizeIterator for $name<'a, T, D, L> {}
        impl<'a, T, D: Dim, L: Layout> FusedIterator for $name<'a, T, D, L> {}

        impl<'a, T, D: Dim, L: Layout> Iterator for $name<'a, T, D, L> {
            type Item = $grid<'a, T, D, L>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index == self.size {
                    None
                } else {
                    let count = self.stride * self.index as isize;

                    self.index += 1;

                    unsafe {
                        Some($grid::new_unchecked(self.ptr.as_ptr().offset(count), self.mapping))
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

impl_axis_iter!(AxisIter, ViewArray, const);
impl_axis_iter!(AxisIterMut, ViewArrayMut, mut);

impl<'a, T, D: Dim, L: Layout> Clone for AxisIter<'a, T, D, L> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            mapping: self.mapping,
            index: self.index,
            size: self.size,
            stride: self.stride,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync, D: Dim, L: Layout> Send for AxisIter<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for AxisIter<'a, T, D, L> {}

unsafe impl<'a, T: Send, D: Dim, L: Layout> Send for AxisIterMut<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for AxisIterMut<'a, T, D, L> {}

macro_rules! impl_flat_iter {
    ($name:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T> $name<'a, T> {
            pub(crate) unsafe fn new_unchecked(
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

        impl<'a, T: Debug> Debug for $name<'a, T> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                struct Value<'b, 'c, U>(&'b $name<'c, U>);

                impl<'b, 'c, U: Debug> Debug for Value<'b, 'c, U> {
                    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                        let mut list = f.debug_list();

                        for i in self.0.index..self.0.size {
                            let count = self.0.stride * i as isize;
                            let _ = list.entry(unsafe { &*self.0.ptr.as_ptr().offset(count) });
                        }

                        list.finish()
                    }
                }

                f.debug_tuple(stringify!($name)).field(&Value(self)).finish()
            }
        }

        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
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

        impl<'a, T> ExactSizeIterator for $name<'a, T> {}
        impl<'a, T> FusedIterator for $name<'a, T> {}

        impl<'a, T> Iterator for $name<'a, T> {
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

impl_flat_iter!(FlatIter, const, {});
impl_flat_iter!(FlatIterMut, mut, {mut});

impl<'a, T> Clone for FlatIter<'a, T> {
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

unsafe impl<'a, T: Sync> Send for FlatIter<'a, T> {}
unsafe impl<'a, T: Sync> Sync for FlatIter<'a, T> {}

unsafe impl<'a, T: Send> Send for FlatIterMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for FlatIterMut<'a, T> {}
