use crate::buffer::OwnedBuffer;
use crate::layout::{Layout, StridedLayout};
use crate::order::Order;
use crate::view::StridedView;
use std::iter::{FusedIterator, TrustedLen};
use std::marker::PhantomData;
use std::ptr;

pub struct Drain<'a, T> {
    start: *const T,
    end: *const T,
    _marker: PhantomData<&'a mut T>,
}

pub struct IntoIter<T, B: OwnedBuffer<T>> {
    iter: B::IntoIter,
}

pub struct Iter<'a, T, const N: usize, const M: usize, O: Order> {
    layout: StridedLayout<N, M, O>,
    start: *const T,
    end: *const T,
    indices: [usize; M],
    inner_size: usize,
    _marker: PhantomData<&'a T>,
}

pub struct IterMut<'a, T, const N: usize, const M: usize, O: Order> {
    layout: StridedLayout<N, M, O>,
    start: *mut T,
    end: *mut T,
    indices: [usize; M],
    inner_size: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Drain<'a, T> {
    pub fn new(ptr: *const T, len: usize) -> Self {
        Self {
            start: ptr,
            end: unsafe { ptr.add(len) },
            _marker: PhantomData,
        }
    }
}

impl<T, B: OwnedBuffer<T>> IntoIter<T, B> {
    pub fn new(iter: B::IntoIter) -> Self {
        Self { iter }
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}
impl<'a, T> FusedIterator for Drain<'a, T> {}

impl<T, B: OwnedBuffer<T>> ExactSizeIterator for IntoIter<T, B> {}
impl<T, B: OwnedBuffer<T>> FusedIterator for IntoIter<T, B> {}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            let current = self.start;

            unsafe {
                self.start = self.start.offset(1);

                Some(ptr::read(current))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { self.end.offset_from(self.start) as usize };

        (len, Some(len))
    }
}

impl<T, B: OwnedBuffer<T>> Iterator for IntoIter<T, B> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

unsafe impl<'a, T> TrustedLen for Drain<'a, T> {}
unsafe impl<T, B: OwnedBuffer<T>> TrustedLen for IntoIter<T, B> {}

macro_rules! impl_iter {
    ($type:ty, $as_ptr:tt, {$($mut:tt)?}) => {
        impl<'a, T, const N: usize, const M: usize, O: Order> $type {
            pub fn new(view: &'a $($mut)? StridedView<T, N, M, O>) -> Self {
                let inner_size = O::select(
                    view.shape()[..N - M].iter().product(),
                    view.shape()[M..].iter().product(),
                );

                Self {
                    layout: view.layout(),
                    start: view.$as_ptr(),
                    end: unsafe { view.$as_ptr().add(inner_size) },
                    indices: [0; M],
                    inner_size,
                    _marker: PhantomData,
                }
            }

            fn outer_size(&self, dim: usize) -> usize {
                O::select(
                    self.layout.shape()[N - M + dim],
                    self.layout.shape()[M - 1 - dim],
                )
            }

            fn outer_stride(&self, dim: usize) -> isize {
                O::select(
                    self.layout.strides()[dim],
                    self.layout.strides()[M - 1 - dim],
                )
            }
        }

        impl<'a, T, const N: usize, const M: usize, O: Order> ExactSizeIterator for $type {}
        impl<'a, T, const N: usize, const M: usize, O: Order> FusedIterator for $type {}

        impl<'a, T, const N: usize, const M: usize, O: Order> Iterator for $type {
            type Item = &'a $($mut)? T;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if self.start == self.end {
                    None
                } else {
                    let current = self.start;

                    unsafe {
                        self.start = self.start.offset(1);

                        if M > 0 && (M == N || self.start == self.end) {
                            self.start = self.start.sub(self.inner_size);
                            self.indices[0] += 1;

                            if self.indices[0] == self.outer_size(0) {
                                self.start = self.start.offset(
                                    (1 - self.outer_size(0) as isize)
                                        * self.outer_stride(0),
                                );
                                self.end = self.start;
                                self.indices[0] = 0;

                                for i in 1..M {
                                    self.indices[i] += 1;

                                    if self.indices[i] == self.outer_size(i) {
                                        self.start = self.start.offset(
                                            (1 - self.outer_size(i) as isize)
                                                * self.outer_stride(i),
                                        );
                                        self.end = self.start;
                                        self.indices[i] = 0;
                                    } else {
                                        self.start = self.start.offset(self.outer_stride(i));
                                        self.end = self.start.add(self.inner_size);
                                        break;
                                    }
                                }
                            } else {
                                self.start = self.start.offset(self.outer_stride(0));
                                self.end = self.start.add(self.inner_size);
                            }
                        }

                        Some(&$($mut)? *current)
                    }
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                if self.start == self.end {
                    (0, Some(0))
                } else {
                    let mut len = unsafe { self.end.offset_from(self.start) as usize };
                    let mut prod = self.inner_size;

                    for i in 0..M {
                        len += (self.outer_size(i) - self.indices[i] - 1) * prod;
                        prod *= self.outer_size(i);
                    }

                    (len, Some(len))
                }
            }
        }

        unsafe impl<'a, T, const N: usize, const M: usize, O: Order> TrustedLen for $type {}
    };
}

impl_iter!(Iter<'a, T, N, M, O>, as_ptr, {});
impl_iter!(IterMut<'a, T, N, M, O>, as_mut_ptr, {mut});
