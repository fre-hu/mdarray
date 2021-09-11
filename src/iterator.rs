use crate::layout::{Layout, StridedLayout};
use crate::order::Order;
use crate::view::StridedView;
use std::iter::FusedIterator;
use std::marker::PhantomData;

macro_rules! impl_iter {
    ($name:tt, $as_ptr:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        pub struct $name<'a, T, const N: usize, const M: usize, O: Order> {
            layout: &'a StridedLayout<N, M, O>,
            start: *$raw_mut T,
            end: *$raw_mut T,
            indices: [usize; M],
            inner_size: usize,
            _marker: PhantomData<&'a $($mut)? T>,
        }

        impl<'a, T, const N: usize, const M: usize, O: Order> $name<'a, T, N, M, O> {
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

        impl<'a, T, const N: usize, const M: usize, O: Order> ExactSizeIterator
            for $name<'a, T, N, M, O>
        {
        }

        impl<'a, T, const N: usize, const M: usize, O: Order> FusedIterator
            for $name<'a, T, N, M, O>
        {
        }

        impl<'a, T, const N: usize, const M: usize, O: Order> Iterator for $name<'a, T, N, M, O> {
            type Item = &'a $($mut)? T;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if self.start == self.end {
                    None
                } else {
                    unsafe {
                        let current = self.start;

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
    };
}

impl_iter!(Iter, as_ptr, const, {});
impl_iter!(IterMut, as_mut_ptr, mut, {mut});
