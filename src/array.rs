use crate::dimension::Dimension;
use crate::index::ViewIndex;
use crate::layout::{DenseLayout, Layout, StridedLayout};
use crate::order::Order;
use crate::raw_vec::RawVec;
use crate::view::ViewBase;
use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::{cmp, mem, ptr};

/// Multidimensional array with static rank and element order.
pub struct ArrayBase<T, B: Buffer<T, N, O>, const N: usize, O: Order> {
    buffer: B,
    _marker: PhantomData<(T, O)>,
}

pub trait Buffer<T, const N: usize, O: Order> {
    type Layout: Layout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_ptr(&self) -> *const T;
    fn layout(&self) -> &Self::Layout;
}

pub struct DenseBuffer<T, const N: usize, O: Order, A: Allocator> {
    vec: RawVec<T, A>,
    layout: DenseLayout<N, O>,
}

pub struct StaticBuffer<T, D: Dimension<N>, const N: usize, O: Order>
where
    [T; D::LEN]: ,
{
    array: [T; D::LEN],
    _marker: PhantomData<(D, O)>,
}

/// Dense multidimensional array with static rank and element order, and dynamic shape.
pub type DenseArray<T, const N: usize, O, A = Global> = ArrayBase<T, DenseBuffer<T, N, O, A>, N, O>;

/// Dense multidimensional array with static rank, shape and element order.
pub type StaticArray<T, D, const N: usize, O> = ArrayBase<T, StaticBuffer<T, D, N, O>, N, O>;

impl<T, const N: usize, O: Order> DenseArray<T, N, O, Global> {
    /// Creates a new, empty array.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T, const N: usize, O: Order, A: Allocator> DenseArray<T, N, O, A> {
    /// Returns a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        self.buffer.vec.allocator()
    }

    /// Returns the number of elements the array can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.buffer.vec.capacity()
    }

    /// Clears the array, removing all values.
    pub fn clear(&mut self) {
        let len = self.len();

        self.buffer.layout.resize([0; N], [0; 0]);

        for i in 0..len {
            unsafe {
                ptr::read(self.buffer.vec.as_ptr().add(i));
            }
        }
    }

    /// Creates a new, empty array with the specified allocator.
    pub fn new_in(alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: DenseBuffer {
                vec: RawVec::new_in(alloc),
                layout: DenseLayout::new([0; N], [0; 0]),
            },
            _marker: PhantomData,
        }
    }

    /// Returns a reshaped array, which must not change the array length.
    pub fn reshape<const M: usize>(self, shape: [usize; M]) -> DenseArray<T, M, O, A> {
        let len = shape
            .iter()
            .fold(1usize, |acc, &x| acc.checked_mul(x).unwrap());

        assert_eq!(len, self.len());

        let me = mem::ManuallyDrop::new(self);

        DenseArray {
            buffer: DenseBuffer {
                vec: unsafe { ptr::read(&me.buffer.vec) },
                layout: DenseLayout::new(shape, [0; 0]),
            },
            _marker: PhantomData,
        }
    }

    /// Shrinks the capacity of the array with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let capacity = cmp::max(min_capacity, self.len());

        if capacity < self.capacity() {
            self.buffer.vec.shrink(capacity);
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    pub fn shrink_to_fit(&mut self) {
        if self.len() < self.capacity() {
            self.buffer.vec.shrink(self.len());
        }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: DenseBuffer {
                vec: RawVec::with_capacity_in(capacity, alloc),
                layout: DenseLayout::new([0; N], [0; 0]),
            },
            _marker: PhantomData,
        }
    }
}

impl<T: Clone, const N: usize, O: Order, A: Allocator> DenseArray<T, N, O, A> {
    /// Resizes the array in-place to the given shape.
    pub fn resize(&mut self, shape: [usize; N], value: T) {
        assert!(self.len() == 0); // TODO: Fix generic resize

        let len = shape
            .iter()
            .fold(1usize, |acc, &x| acc.checked_mul(x).unwrap());

        if len > self.capacity() {
            self.buffer.vec.grow(len);
        }

        for i in 0..len {
            unsafe {
                ptr::write(self.buffer.vec.as_mut_ptr().add(i), value.clone());
            }
        }

        self.buffer.layout.resize(shape, [0; 0]);
    }
}

impl<T: Copy, D: Dimension<N>, const N: usize, O: Order> StaticArray<T, D, N, O>
where
    [(); D::LEN]: ,
{
    /// Creates a new array, where the given value is copied to each element.
    pub fn new(value: T) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: StaticBuffer {
                array: [value; D::LEN],
                _marker: PhantomData,
            },
            _marker: PhantomData,
        }
    }
}

impl<T, D: Dimension<N>, const N: usize, O: Order> StaticBuffer<T, D, N, O>
where
    [T; D::LEN]: ,
{
    const LAYOUT: DenseLayout<N, O> = DenseLayout::new(D::SHAPE, [0; 0]);
}

impl<T, const N: usize, O: Order, A: Allocator> Buffer<T, N, O> for DenseBuffer<T, N, O, A> {
    type Layout = DenseLayout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }

    fn layout(&self) -> &DenseLayout<N, O> {
        &self.layout
    }
}

impl<T, const N: usize, O: Order, D: Dimension<N>> Buffer<T, N, O> for StaticBuffer<T, D, N, O>
where
    [(); D::LEN]: ,
{
    type Layout = DenseLayout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.array.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.array.as_ptr()
    }

    fn layout(&self) -> &DenseLayout<N, O> {
        &Self::LAYOUT
    }
}

impl<T, B: Buffer<T, N, O> + Clone, const N: usize, O: Order> Clone for ArrayBase<T, B, N, O> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Clone, const N: usize, O: Order, A: Allocator + Clone> Clone for DenseBuffer<T, N, O, A> {
    fn clone(&self) -> Self {
        let len = self.layout.shape().iter().product();

        let mut vec = RawVec::<T, A>::with_capacity_in(len, self.vec.allocator().clone());

        for i in 0..len {
            unsafe {
                ptr::write(vec.as_mut_ptr().add(i), (*self.vec.as_ptr().add(i)).clone());
            }
        }

        Self {
            vec,
            layout: DenseLayout::new(self.layout.shape().clone(), [0; 0]),
        }
    }
}

impl<T: Clone, const N: usize, O: Order, D: Dimension<N>> Clone for StaticBuffer<T, D, N, O>
where
    [(); D::LEN]: ,
{
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, O: Order> Default for DenseArray<T, N, O, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, D: Dimension<N>, const N: usize, O: Order> Default
    for StaticArray<T, D, N, O>
where
    [(); D::LEN]: ,
{
    fn default() -> Self {
        Self {
            buffer: StaticBuffer {
                array: [T::default(); D::LEN], // TODO: Change to Default::default()
                _marker: PhantomData,
            },
            _marker: PhantomData,
        }
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> Deref for ArrayBase<T, B, N, O> {
    type Target = ViewBase<T, B::Layout, N, O>;

    fn deref(&self) -> &Self::Target {
        unsafe { ViewBase::from_raw_parts(self.buffer.as_ptr(), self.buffer.layout()) }
    }
}

impl<T, B: Buffer<T, N, O>, const N: usize, O: Order> DerefMut for ArrayBase<T, B, N, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { ViewBase::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.layout()) }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> Drop for DenseBuffer<T, N, O, A> {
    fn drop(&mut self) {
        for i in 0..self.layout.shape().iter().product() {
            unsafe {
                ptr::read(self.vec.as_ptr().add(i));
            }
        }
    }
}

impl<I: ViewIndex<T, N, M, O>, T, B, const N: usize, const M: usize, O: Order> Index<I>
    for ArrayBase<T, B, N, O>
where
    B: Buffer<T, N, O, Layout = StridedLayout<N, M, O>>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(&**self)
    }
}

impl<I: ViewIndex<T, N, M, O>, T, B, const N: usize, const M: usize, O: Order> IndexMut<I>
    for ArrayBase<T, B, N, O>
where
    B: Buffer<T, N, O, Layout = StridedLayout<N, M, O>>,
{
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(&mut **self)
    }
}
