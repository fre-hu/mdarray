use crate::iterator::{ViewIter, ViewIterMut};
use crate::layout::{DenseLayout, Layout, StridedLayout};
use crate::order::Order;
use std::marker::PhantomData;
use std::ops::{
    Bound, Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use std::slice::{self, SliceIndex};

/// Multidimensional view into an array with static rank and element order.
#[repr(transparent)]
pub struct ViewBase<T, L: Layout<N, M>, const N: usize, const M: usize> {
    _data: PhantomData<T>,
    _layout: PhantomData<L>,
    _slice: [()],
}

pub trait SliceOrViewIndex<T, L: Layout<N, M>, const N: usize, const M: usize> {
    type Output: ?Sized;

    fn index(self, view: &ViewBase<T, L, N, M>) -> &Self::Output;
    fn index_mut(self, view: &mut ViewBase<T, L, N, M>) -> &mut Self::Output;
}

/// Multidimensional view with static rank and element order, and dynamic shape and strides.
pub type StridedView<T, const N: usize, const M: usize, const O: Order> =
    ViewBase<T, StridedLayout<N, M, O>, N, M>;

/// Dense multidimensional view with static rank and element order, and dynamic shape.
pub type DenseView<T, const N: usize, const O: Order> = ViewBase<T, StridedLayout<N, 0, O>, N, 0>;

impl<T, L: DenseLayout<N>, const N: usize> ViewBase<T, L, N, 0> {
    /// Returns a mutable slice of all elements in the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns a slice of all elements in the array.
    pub fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T, L: Layout<N, M>, const N: usize, const M: usize> ViewBase<T, L, N, M> {
    /// Returns an iterator over the array.
    pub fn iter(&self) -> ViewIter<'_, T, L, N, M> {
        ViewIter::new(self)
    }

    /// Returns a mutable iterator over the array.
    pub fn iter_mut(&mut self) -> ViewIterMut<'_, T, L, N, M> {
        ViewIterMut::new(self)
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns the number of dimensions of the array.
    pub fn rank(&self) -> usize {
        N
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize; N] {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
    }

    /// Returns the distance between elements in the specified dimension.
    pub fn stride(&self, dim: usize) -> isize {
        self.layout().stride(dim)
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut T {
        let (data, _) = (self as *const Self).to_raw_parts();

        data as *mut T
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        let (data, _) = (self as *const Self).to_raw_parts();

        data as *const T
    }

    pub(crate) fn layout(&self) -> &L {
        let (_, layout) = (self as *const Self).to_raw_parts();

        unsafe { &*(layout as *const L) }
    }
}

impl<T, L: DenseLayout<N>, const N: usize> Deref for ViewBase<T, L, N, 0> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, L: DenseLayout<N>, const N: usize> DerefMut for ViewBase<T, L, N, 0> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl<T, L: Layout<N, M>, I: SliceOrViewIndex<T, L, N, M>, const N: usize, const M: usize> Index<I>
    for ViewBase<T, L, N, M>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        SliceOrViewIndex::index(index, self)
    }
}

impl<T, L: Layout<N, M>, I: SliceOrViewIndex<T, L, N, M>, const N: usize, const M: usize>
    IndexMut<I> for ViewBase<T, L, N, M>
{
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        SliceOrViewIndex::index_mut(index, self)
    }
}

macro_rules! impl_slice_index {
    ($t:ty) => {
        impl<T, L: DenseLayout<N>, const N: usize> SliceOrViewIndex<T, L, N, 0> for $t {
            type Output = <$t as SliceIndex<[T]>>::Output;

            fn index(self, view: &ViewBase<T, L, N, 0>) -> &Self::Output {
                Index::index(view.deref(), self)
            }

            fn index_mut(self, view: &mut ViewBase<T, L, N, 0>) -> &mut Self::Output {
                IndexMut::index_mut(view.deref_mut(), self)
            }
        }
    };
}

impl_slice_index!((Bound<usize>, Bound<usize>));
impl_slice_index!(usize);
impl_slice_index!(Range<usize>);
impl_slice_index!(RangeFrom<usize>);
impl_slice_index!(RangeInclusive<usize>);
impl_slice_index!(RangeFull);
impl_slice_index!(RangeTo<usize>);
impl_slice_index!(RangeToInclusive<usize>);

impl<T, L: Layout<N, M>, const N: usize, const M: usize> SliceOrViewIndex<T, L, N, M>
    for [usize; N]
{
    type Output = T;

    fn index(self, view: &ViewBase<T, L, N, M>) -> &Self::Output {
        let index = self
            .iter()
            .enumerate()
            .map(|(i, &x)| x as isize * view.stride(i))
            .sum();

        unsafe { &*view.as_ptr().offset(index) }
    }

    fn index_mut(self, view: &mut ViewBase<T, L, N, M>) -> &mut Self::Output {
        let index = self
            .iter()
            .enumerate()
            .map(|(i, &x)| x as isize * view.stride(i))
            .sum();

        unsafe { &mut *view.as_mut_ptr().offset(index) }
    }
}
