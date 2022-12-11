use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::dim::Dim;
use crate::format::Format;
use crate::grid::{SubGrid, SubGridMut};
use crate::layout::Layout;
use crate::span::SpanBase;

/// Array axis iterator.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct AxisIter<'a, T, D: Dim, F: Format> {
    ptr: NonNull<T>,
    layout: Layout<D, F>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

/// Mutable array axis iterator.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct AxisIterMut<'a, T, D: Dim, F: Format> {
    ptr: NonNull<T>,
    layout: Layout<D, F>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

/// Linear array span iterator.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct LinearIter<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

/// Mutable linear array span iterator.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct LinearIterMut<'a, T> {
    ptr: NonNull<T>,
    index: usize,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

macro_rules! impl_axis_iter {
    ($name:tt, $grid:tt, $raw_mut:tt) => {
        impl<'a, T, D: Dim, F: Format> $name<'a, T, D, F> {
            pub(crate) unsafe fn new_unchecked(
                ptr: *$raw_mut T,
                layout: Layout<D, F>,
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

        impl<'a, T, D: Dim, F: Format> Debug for $name<'a, T, D, F>
        where
            SpanBase<T, D, F>: Debug,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                struct Value<'b, 'c, U, E: Dim, G: Format>(&'b $name<'c, U, E, G>);

                impl<'b, 'c, U, E: Dim, G: Format> Debug for Value<'b, 'c, U, E, G>
                where
                    SpanBase<U, E, G>: Debug,
                {
                    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                        let mut list = f.debug_list();

                        for i in self.0.index..self.0.size {
                            unsafe {
                                let ptr = self.0.ptr.as_ptr().offset(self.0.stride * i as isize);

                                list.entry(&SubGrid::new_unchecked(ptr, self.0.layout));
                            }
                        }

                        list.finish()
                    }
                }

                f.debug_tuple(stringify!($name)).field(&Value(self)).finish()
            }
        }

        impl<'a, T, D: Dim, F: Format> DoubleEndedIterator for $name<'a, T, D, F> {
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

        impl<'a, T, D: Dim, F: Format> ExactSizeIterator for $name<'a, T, D, F> {}
        impl<'a, T, D: Dim, F: Format> FusedIterator for $name<'a, T, D, F> {}

        impl<'a, T, D: Dim, F: Format> Iterator for $name<'a, T, D, F> {
            type Item = $grid<'a, T, D, F>;

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

impl_axis_iter!(AxisIter, SubGrid, const);
impl_axis_iter!(AxisIterMut, SubGridMut, mut);

impl<'a, T, D: Dim, F: Format> Clone for AxisIter<'a, T, D, F> {
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

unsafe impl<'a, T: Sync, D: Dim, F: Format> Send for AxisIter<'a, T, D, F> {}
unsafe impl<'a, T: Sync, D: Dim, F: Format> Sync for AxisIter<'a, T, D, F> {}

unsafe impl<'a, T: Send, D: Dim, F: Format> Send for AxisIterMut<'a, T, D, F> {}
unsafe impl<'a, T: Sync, D: Dim, F: Format> Sync for AxisIterMut<'a, T, D, F> {}

macro_rules! impl_linear_iter {
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

                            list.entry(unsafe { &*self.0.ptr.as_ptr().offset(count) });
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

impl_linear_iter!(LinearIter, const, {});
impl_linear_iter!(LinearIterMut, mut, {mut});

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
