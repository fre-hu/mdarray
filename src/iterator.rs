use crate::layout::Layout;
use crate::order::Order;
use crate::view::ViewBase;
use std::iter::Iterator;
use std::marker::PhantomData;

trait ShapedIterator<const N: usize, const M: usize>: Iterator {
    const ORDER: Order;

    fn shape(&self) -> &[usize; N];
    fn size(&self, dim: usize) -> usize;
}

pub struct ViewIter<'a, T, L: Layout<N, M>, const N: usize, const M: usize> {
    layout: &'a L,
    start: *const T,
    end: *const T,
    indices: [usize; M],
    _data: PhantomData<&'a T>,
}

pub struct ViewIterMut<'a, T, L: Layout<N, M>, const N: usize, const M: usize> {
    layout: &'a L,
    start: *mut T,
    end: *mut T,
    indices: [usize; M],
    _data: PhantomData<&'a T>,
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> ViewIter<'a, T, L, N, M> {
    pub fn new(view: &'a ViewBase<T, L, N, M>) -> Self {
        Self {
            layout: view.layout(),
            start: view.as_ptr(),
            end: unsafe { view.as_ptr().add(view.layout().inner_len()) },
            indices: [0; M],
            _data: PhantomData,
        }
    }
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> Iterator for ViewIter<'a, T, L, N, M> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let current = self.start;

                self.start = self.start.offset(1);

                if M > 0 && (M == N || self.start == self.end) {
                    self.start = self.start.sub(self.layout.inner_len());
                    self.end = self.start;

                    for i in 0..M {
                        self.indices[i] += 1;

                        if self.indices[i] == self.layout.outer_size(i) {
                            self.start = self.start.offset(
                                (1 - self.layout.outer_size(i) as isize)
                                    * self.layout.outer_stride(i),
                            );
                            self.end = self.start;
                            self.indices[i] = 0;
                        } else {
                            self.start = self.start.offset(self.layout.outer_stride(i));
                            self.end = self.start.add(self.layout.inner_len());
                            break;
                        }
                    }
                }

                Some(&*current)
            }
        }
    }
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> ShapedIterator<N, M>
    for ViewIter<'a, T, L, N, M>
{
    const ORDER: Order = L::ORDER;

    fn shape(&self) -> &[usize; N] {
        self.layout.shape()
    }

    fn size(&self, dim: usize) -> usize {
        self.layout.size(dim)
    }
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> ViewIterMut<'a, T, L, N, M> {
    pub fn new(view: &'a mut ViewBase<T, L, N, M>) -> Self {
        Self {
            layout: view.layout(),
            start: view.as_mut_ptr(),
            end: unsafe { view.as_mut_ptr().add(view.layout().inner_len()) },
            indices: [0; M],
            _data: PhantomData,
        }
    }
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> Iterator
    for ViewIterMut<'a, T, L, N, M>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let current = self.start;

                self.start = self.start.offset(1);

                if M > 0 && (M == N || self.start == self.end) {
                    self.start = self.start.sub(self.layout.inner_len());
                    self.end = self.start;

                    for i in 0..M {
                        self.indices[i] += 1;

                        if self.indices[i] == self.layout.outer_size(i) {
                            self.start = self.start.offset(
                                (1 - self.layout.outer_size(i) as isize)
                                    * self.layout.outer_stride(i),
                            );
                            self.end = self.start;
                            self.indices[i] = 0;
                        } else {
                            self.start = self.start.offset(self.layout.outer_stride(i));
                            self.end = self.start.add(self.layout.inner_len());
                            break;
                        }
                    }
                }

                Some(&mut *current)
            }
        }
    }
}

impl<'a, T, L: Layout<N, M>, const N: usize, const M: usize> ShapedIterator<N, M>
    for ViewIterMut<'a, T, L, N, M>
{
    const ORDER: Order = L::ORDER;

    fn shape(&self) -> &[usize; N] {
        self.layout.shape()
    }

    fn size(&self, dim: usize) -> usize {
        self.layout.size(dim)
    }
}
