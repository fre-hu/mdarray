use crate::aligned_alloc::AlignedAlloc;
use crate::buffer::{Buffer, FromIterIn};
use crate::buffer::{DenseBuffer, StaticBuffer, SubBuffer, SubBufferMut};
use crate::dimension::{Dim1, Dim2, Dimension};
use crate::iterator::{Drain, IntoIter};
use crate::layout::StridedLayout;
use crate::order::{ColumnMajor, Order, RowMajor};
use crate::view::{StridedView, ViewBase};
use std::alloc::{Allocator, Global};
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Result};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::{cmp, mem};

/// Multidimensional array with static rank and element order.
pub struct GridBase<T, B: Buffer<T, N, O>, const N: usize, O: Order> {
    buffer: B,
    _marker: PhantomData<(T, O)>,
}

/// Dense multidimensional array with static rank and element order, and dynamic shape.
pub type DenseGrid<T, const N: usize, O, A = AlignedAlloc> =
    GridBase<T, DenseBuffer<T, N, O, A>, N, O>;

/// Dense multidimensional array with static rank, shape and element order.
pub type StaticGrid<T, D, const N: usize, O> = GridBase<T, StaticBuffer<T, D, N, O>, N, O>;

pub type SubGrid<'a, T, const N: usize, const M: usize, O> =
    GridBase<T, SubBuffer<'a, T, N, M, O>, N, O>;

pub type SubGridMut<'a, T, const N: usize, const M: usize, O> =
    GridBase<T, SubBufferMut<'a, T, N, M, O>, N, O>;

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> GridBase<T, B, N, O> {
    /// Returns a mutable array view of the entire array.
    pub fn as_mut_view(&mut self) -> &mut ViewBase<T, B::Layout, N, O> {
        unsafe { ViewBase::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.layout()) }
    }

    /// Returns an array view of the entire array.
    pub fn as_view(&self) -> &ViewBase<T, B::Layout, N, O> {
        unsafe { ViewBase::from_raw_parts(self.buffer.as_ptr(), self.buffer.layout()) }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> DenseGrid<T, N, O, A> {
    /// Returns a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        self.buffer.allocator()
    }

    /// Returns the number of elements the array can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Clears the array, removing all values.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Creates a draining iterator over all elemets in the array.
    pub fn drain(&mut self) -> Drain<T> {
        self.buffer.drain()
    }

    /// Creates an array from raw components of another array with the specified allocator.
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: [usize; N],
        capacity: usize,
        alloc: A,
    ) -> Self {
        Self {
            buffer: DenseBuffer::from_raw_parts_in(ptr, shape, capacity, alloc),
            _marker: PhantomData,
        }
    }

    /// Decomposes an array into its raw components including the allocator.
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, [usize; N], usize, A) {
        let Self { buffer, .. } = self;

        buffer.into_raw_parts_with_alloc()
    }

    /// Creates a new, empty array with the specified allocator.
    pub fn new_in(alloc: A) -> Self {
        Self {
            buffer: DenseBuffer::new_in(alloc),
            _marker: PhantomData,
        }
    }

    /// Returns a reshaped array, which must not change the array length.
    pub fn reshape<const M: usize>(self, shape: [usize; M]) -> DenseGrid<T, M, O, A> {
        let len = shape.iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert_eq!(len, self.len());

        let (ptr, _, capacity, alloc) = self.into_raw_parts_with_alloc();

        DenseGrid {
            buffer: unsafe { DenseBuffer::from_raw_parts_in(ptr, shape, capacity, alloc) },
            _marker: PhantomData,
        }
    }

    /// Shrinks the capacity of the array with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.buffer.shrink_to(cmp::max(min_capacity, self.len()));
    }

    /// Shrinks the capacity of the array as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.buffer.shrink_to(self.len());
    }

    /// Converts the array into a vector.
    pub fn into_vec(self) -> Vec<T, A> {
        let (ptr, shape, capacity, alloc) = self.into_raw_parts_with_alloc();

        unsafe { Vec::from_raw_parts_in(ptr, shape.iter().product(), capacity, alloc) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            buffer: DenseBuffer::with_capacity_in(capacity, alloc),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, O: Order> DenseGrid<T, N, O, AlignedAlloc> {
    /// Creates an array from raw components of another array.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: [usize; N], capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, shape, capacity, AlignedAlloc::new(Global))
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, [usize; N], usize) {
        let (ptr, shape, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, shape, capacity)
    }

    /// Creates a new, empty array.
    pub fn new() -> Self {
        Self::new_in(AlignedAlloc::new(Global))
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, AlignedAlloc::new(Global))
    }
}

impl<T: Clone, const N: usize, O: Order, A: Allocator> DenseGrid<T, N, O, A> {
    /// Resizes the array in-place to the given shape.
    pub fn resize(&mut self, shape: [usize; N], value: T) {
        self.buffer.resize(shape, value);
    }
}

impl<T: Copy, D: Dimension<N>, const N: usize, O: Order> StaticGrid<T, D, N, O>
where
    [(); D::LEN]: ,
{
    /// Creates a new array, where the given value is copied to each element.
    pub fn new(value: T) -> Self {
        Self {
            buffer: StaticBuffer::new(value), // TODO: Change to [value; D::LEN]
            _marker: PhantomData,
        }
    }
}

impl<T, const X: usize, O: Order> StaticGrid<T, Dim1<X>, 1, O>
where
    [(); Dim1::<X>::LEN]: ,
{
    /// Converts the array into a primitive array.
    pub fn into_array(self) -> [T; X] {
        let grid = mem::ManuallyDrop::new(self);

        unsafe { ptr::read(grid.as_ptr() as *const [T; X]) }
    }
}

impl<T, const X: usize, const Y: usize> StaticGrid<T, Dim2<X, Y>, 2, ColumnMajor>
where
    [(); Dim2::<X, Y>::LEN]: ,
{
    /// Converts the array into a primitive array.
    pub fn into_array(self) -> [[T; X]; Y] {
        let grid = mem::ManuallyDrop::new(self);

        unsafe { ptr::read(grid.as_ptr() as *const [[T; X]; Y]) }
    }
}

impl<T, const X: usize, const Y: usize> StaticGrid<T, Dim2<X, Y>, 2, RowMajor>
where
    [(); Dim2::<X, Y>::LEN]: ,
{
    /// Converts the array into a primitive array.
    pub fn into_array(self) -> [[T; Y]; X] {
        let grid = mem::ManuallyDrop::new(self);

        unsafe { ptr::read(grid.as_ptr() as *const [[T; Y]; X]) }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubGrid<'a, T, N, M, O> {
    /// Creates a subarray from the specified pointer and layout.
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        Self {
            buffer: SubBuffer::new(ptr, layout),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubGridMut<'a, T, N, M, O> {
    /// Creates a mutable subarray from the specified pointer and layout.
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        Self {
            buffer: SubBufferMut::new(ptr, layout),
            _marker: PhantomData,
        }
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> Borrow<ViewBase<T, B::Layout, N, O>>
    for GridBase<T, B, N, O>
{
    fn borrow(&self) -> &ViewBase<T, B::Layout, N, O> {
        self.as_view()
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> BorrowMut<ViewBase<T, B::Layout, N, O>>
    for GridBase<T, B, N, O>
{
    fn borrow_mut(&mut self) -> &mut ViewBase<T, B::Layout, N, O> {
        self.as_mut_view()
    }
}

impl<T, B: Buffer<T, N, O> + Clone, const N: usize, O: Order> Clone for GridBase<T, B, N, O> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> Debug for GridBase<T, B, N, O>
where
    ViewBase<T, B::Layout, N, O>: Debug,
{
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        self.as_view().fmt(fmt)
    }
}

impl<T, const N: usize, O: Order> Default for DenseGrid<T, N, O, AlignedAlloc> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, D: Dimension<N>, const N: usize, O: Order> Default
    for StaticGrid<T, D, N, O>
where
    [(); D::LEN]: ,
{
    fn default() -> Self {
        Self::new(Default::default()) // TODO: Change to array and remove T: Copy
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> Deref for GridBase<T, B, N, O> {
    type Target = ViewBase<T, B::Layout, N, O>;

    fn deref(&self) -> &Self::Target {
        self.as_view()
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> DerefMut for GridBase<T, B, N, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_view()
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> Eq for GridBase<T, B, N, O> where
    ViewBase<T, B::Layout, N, O>: Eq
{
}

impl<T, O: Order, A: Allocator> From<Vec<T, A>> for DenseGrid<T, 1, O, A> {
    fn from(vec: Vec<T, A>) -> Self {
        let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_alloc();

        unsafe { Self::from_raw_parts_in(ptr, [len], capacity, alloc) }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> From<DenseGrid<T, N, O, A>> for Vec<T, A> {
    fn from(grid: DenseGrid<T, N, O, A>) -> Self {
        grid.into_vec()
    }
}

// TODO: Add From<[..]> using StaticGrid::new(array)

impl<T, const X: usize, O: Order> From<StaticGrid<T, Dim1<X>, 1, O>> for [T; X]
where
    [(); Dim1::<X>::LEN]: ,
{
    fn from(grid: StaticGrid<T, Dim1<X>, 1, O>) -> Self {
        grid.into_array()
    }
}

impl<T, const X: usize, const Y: usize> From<StaticGrid<T, Dim2<X, Y>, 2, ColumnMajor>>
    for [[T; X]; Y]
where
    [(); Dim2::<X, Y>::LEN]: ,
{
    fn from(grid: StaticGrid<T, Dim2<X, Y>, 2, ColumnMajor>) -> Self {
        grid.into_array()
    }
}

impl<T, const X: usize, const Y: usize> From<StaticGrid<T, Dim2<X, Y>, 2, RowMajor>> for [[T; Y]; X]
where
    [(); Dim2::<X, Y>::LEN]: ,
{
    fn from(grid: StaticGrid<T, Dim2<X, Y>, 2, RowMajor>) -> Self {
        grid.into_array()
    }
}

impl<T, O: Order> FromIterator<T> for DenseGrid<T, 1, O> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            buffer: DenseBuffer::from_iter_in(iter.into_iter(), AlignedAlloc::new(Global)),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, B: Buffer<T, N, O>, const N: usize, O: Order> IntoIterator for &'a GridBase<T, B, N, O>
where
    &'a ViewBase<T, B::Layout, N, O>: IntoIterator<Item = &'a T>,
{
    type Item = &'a T;
    type IntoIter = <&'a ViewBase<T, B::Layout, N, O> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_view().into_iter()
    }
}

impl<'a, T, B: Buffer<T, N, O>, const N: usize, O: Order> IntoIterator
    for &'a mut GridBase<T, B, N, O>
where
    &'a mut ViewBase<T, B::Layout, N, O>: IntoIterator<Item = &'a mut T>,
{
    type Item = &'a mut T;
    type IntoIter = <&'a mut ViewBase<T, B::Layout, N, O> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_view().into_iter()
    }
}

impl<T, const N: usize, O: Order, A: Allocator> IntoIterator for DenseGrid<T, N, O, A> {
    type Item = T;
    type IntoIter = IntoIter<T, DenseBuffer<T, N, O, A>>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(Vec::from(self).into_iter())
    }
}

impl<T, D: Dimension<N>, const N: usize, O: Order> IntoIterator for StaticGrid<T, D, N, O>
where
    [(); D::LEN]: ,
{
    type Item = T;
    type IntoIter = IntoIter<T, StaticBuffer<T, D, N, O>>;

    fn into_iter(self) -> Self::IntoIter {
        let me = mem::ManuallyDrop::new(self);
        let array = unsafe { ptr::read(me.as_ptr() as *const [T; D::LEN]) };

        Self::IntoIter::new(<[T; D::LEN] as IntoIterator>::into_iter(array))
    }
}

impl<T, B: Buffer<T, 1, O>, O: Order> Ord for GridBase<T, B, 1, O>
where
    ViewBase<T, B::Layout, 1, O>: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(other.as_view())
    }
}

impl<T, U, B: Buffer<T, N, O>, C: Buffer<U, N, O>, const N: usize, O: Order>
    PartialEq<GridBase<U, C, N, O>> for GridBase<T, B, N, O>
where
    ViewBase<T, B::Layout, N, O>: PartialEq<ViewBase<U, C::Layout, N, O>>,
{
    fn eq(&self, other: &GridBase<U, C, N, O>) -> bool {
        self.as_view().eq(other.as_view())
    }
}

impl<T, U, B: Buffer<T, N, O>, const N: usize, const M: usize, O: Order>
    PartialEq<StridedView<U, N, M, O>> for GridBase<T, B, N, O>
where
    ViewBase<T, B::Layout, N, O>: PartialEq<StridedView<U, N, M, O>>,
{
    fn eq(&self, other: &StridedView<U, N, M, O>) -> bool {
        self.as_view().eq(other)
    }
}

impl<T, U, B: Buffer<U, N, O>, const N: usize, const M: usize, O: Order>
    PartialEq<GridBase<U, B, N, O>> for StridedView<T, N, M, O>
where
    StridedView<T, N, M, O>: PartialEq<ViewBase<U, B::Layout, N, O>>,
{
    fn eq(&self, other: &GridBase<U, B, N, O>) -> bool {
        self.eq(other.as_view())
    }
}

impl<T, U, B: Buffer<T, 1, O>, C: Buffer<U, 1, O>, O: Order> PartialOrd<GridBase<U, C, 1, O>>
    for GridBase<T, B, 1, O>
where
    ViewBase<T, B::Layout, 1, O>: PartialOrd<ViewBase<U, C::Layout, 1, O>>,
{
    fn partial_cmp(&self, other: &GridBase<U, C, 1, O>) -> Option<Ordering> {
        self.as_view().partial_cmp(other.as_view())
    }
}

impl<T, U, B: Buffer<T, 1, O>, const M: usize, O: Order> PartialOrd<StridedView<U, 1, M, O>>
    for GridBase<T, B, 1, O>
where
    ViewBase<T, B::Layout, 1, O>: PartialOrd<StridedView<U, 1, M, O>>,
{
    fn partial_cmp(&self, other: &StridedView<U, 1, M, O>) -> Option<Ordering> {
        self.as_view().partial_cmp(other)
    }
}

impl<T, U, B: Buffer<U, 1, O>, const M: usize, O: Order> PartialOrd<GridBase<U, B, 1, O>>
    for StridedView<T, 1, M, O>
where
    StridedView<T, 1, M, O>: PartialOrd<ViewBase<U, B::Layout, 1, O>>,
{
    fn partial_cmp(&self, other: &GridBase<U, B, 1, O>) -> Option<Ordering> {
        self.partial_cmp(other.as_view())
    }
}
