use crate::index::ViewIndex;
use crate::layout::StridedLayout;
use crate::order::Order;
use crate::view::StridedView;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

pub struct SubArray<'a, T, const N: usize, const M: usize, O: Order> {
    ptr: NonNull<T>,
    layout: StridedLayout<N, M, O>,
    _marker: PhantomData<&'a T>,
}

pub struct SubArrayMut<'a, T, const N: usize, const M: usize, O: Order> {
    ptr: NonNull<T>,
    layout: StridedLayout<N, M, O>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubArray<'a, T, N, M, O> {
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        Self {
            ptr,
            layout,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubArrayMut<'a, T, N, M, O> {
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        Self {
            ptr,
            layout,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> Deref for SubArray<'a, T, N, M, O> {
    type Target = StridedView<T, N, M, O>;

    fn deref(&self) -> &Self::Target {
        unsafe { StridedView::from_raw_parts(self.ptr.as_ptr(), &self.layout) }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> Deref for SubArrayMut<'a, T, N, M, O> {
    type Target = StridedView<T, N, M, O>;

    fn deref(&self) -> &Self::Target {
        unsafe { StridedView::from_raw_parts(self.ptr.as_ptr(), &self.layout) }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> DerefMut for SubArrayMut<'a, T, N, M, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { StridedView::from_raw_parts_mut(self.ptr.as_ptr(), &self.layout) }
    }
}

impl<'a, I: ViewIndex<T, N, M, O>, T, const N: usize, const M: usize, O: Order> Index<I>
    for SubArray<'a, T, N, M, O>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(&**self)
    }
}

impl<'a, I: ViewIndex<T, N, M, O>, T, const N: usize, const M: usize, O: Order> Index<I>
    for SubArrayMut<'a, T, N, M, O>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(&**self)
    }
}

impl<'a, I: ViewIndex<T, N, M, O>, T, const N: usize, const M: usize, O: Order> IndexMut<I>
    for SubArrayMut<'a, T, N, M, O>
{
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(&mut **self)
    }
}
