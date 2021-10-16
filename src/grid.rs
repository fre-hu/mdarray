use crate::buffer::Buffer;
use crate::buffer::{DenseBuffer, StaticBuffer, SubBuffer, SubBufferMut};
use crate::dimension::Dimension;
use crate::layout::StridedLayout;
use crate::order::Order;
use crate::view::ViewBase;
use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::{cmp, mem};

/// Multidimensional array with static rank and element order.
pub struct GridBase<T, B: Buffer<T, N, O>, const N: usize, O: Order> {
    buffer: B,
    _marker: PhantomData<(T, O)>,
}

/// Dense multidimensional array with static rank and element order, and dynamic shape.
pub type DenseGrid<T, const N: usize, O, A = Global> = GridBase<T, DenseBuffer<T, N, O, A>, N, O>;

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

    /// Creates an array from raw components of another array with the specified allocator.
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: [usize; N],
        capacity: usize,
        alloc: A,
    ) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

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
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

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

    /// Creates a new, empty array with the specified capacity and allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: DenseBuffer::with_capacity_in(capacity, alloc),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, O: Order> DenseGrid<T, N, O, Global> {
    /// Creates an array from raw components of another array.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: [usize; N], capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, shape, capacity, Global)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, [usize; N], usize) {
        let (ptr, shape, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, shape, capacity)
    }

    /// Creates a new, empty array.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
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
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: StaticBuffer::new(value), // TODO: Change to [value; D::LEN]
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubGrid<'a, T, N, M, O> {
    /// Creates a subarray from the specified pointer and layout.
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: SubBuffer::new(ptr, layout),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubGridMut<'a, T, N, M, O> {
    /// Creates a mutable subarray from the specified pointer and layout.
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            buffer: SubBufferMut::new(ptr, layout),
            _marker: PhantomData,
        }
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

impl<T, const N: usize, O: Order> Default for DenseGrid<T, N, O, Global> {
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
