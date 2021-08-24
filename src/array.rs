use crate::dimension::Dimension;
use crate::layout::{Layout, StaticLayout, StridedLayout};
use crate::order::Order;
use crate::view::ViewBase;
use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// Multidimensional array with static rank and element order.
pub struct ArrayBase<T, B: Buffer<T>, L: Layout<N, 0>, const N: usize> {
    buffer: B,
    layout: L,
    _data: PhantomData<T>,
}

pub trait Buffer<T> {
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub struct DenseBuffer<T, A: Allocator> {
    vec: Vec<T, A>, // TODO: Replace with RawVec<T>
}

pub struct StaticBuffer<T, const L: usize> {
    array: [T; L],
}

/// Dense multidimensional array with static rank and element order, and dynamic shape.
pub type DenseArray<T, A: Allocator, const N: usize, const O: Order> =
    ArrayBase<T, DenseBuffer<T, A>, StridedLayout<N, 0, O>, N>;

/// Multidimensional array with static rank, shape and element order.
pub type StaticArray<T, D: Dimension<N>, const N: usize, const O: Order> =
    ArrayBase<T, StaticBuffer<T, { D::LEN }>, StaticLayout<D, N, O>, N>;

impl<T, const N: usize, const O: Order>
    ArrayBase<T, DenseBuffer<T, Global>, StridedLayout<N, 0, O>, N>
{
    /// Constructs a new, empty array.
    pub fn new() -> Self {
        Self::new_in(Global)
    }
}

impl<T: Copy, D: Dimension<N>, const N: usize, const O: Order>
    ArrayBase<T, StaticBuffer<T, { D::LEN }>, StaticLayout<D, N, O>, N>
{
    /// Constructs a new array, where the given value is copied to each element.
    pub fn new(value: T) -> Self {
        Self {
            buffer: StaticBuffer {
                array: [value; D::LEN],
            },
            layout: StaticLayout::new(),
            _data: PhantomData,
        }
    }
}

impl<T, A: Allocator, const N: usize, const O: Order>
    ArrayBase<T, DenseBuffer<T, A>, StridedLayout<N, 0, O>, N>
{
    /// Constructs a new, empty array with the specified allocator.
    pub fn new_in(alloc: A) -> Self {
        Self {
            buffer: DenseBuffer {
                vec: Vec::new_in(alloc),
            },
            layout: StridedLayout::new([0; N], [0; 0]),
            _data: PhantomData,
        }
    }

    /// Returns a reshaped array, which must not change the array length.
    pub fn reshape<const M: usize>(
        self,
        shape: [usize; M],
    ) -> ArrayBase<T, DenseBuffer<T, A>, StridedLayout<M, 0, O>, M> {
        assert_eq!(shape.iter().product::<usize>(), self.layout.len());

        let me = mem::ManuallyDrop::new(self);

        ArrayBase::<T, DenseBuffer<T, A>, StridedLayout<M, 0, O>, M> {
            buffer: unsafe { ptr::read(&me.buffer) },
            layout: StridedLayout::new(shape, [0; 0]),
            _data: PhantomData,
        }
    }
}

impl<T: Clone, A: Allocator, const N: usize, const O: Order>
    ArrayBase<T, DenseBuffer<T, A>, StridedLayout<N, 0, O>, N>
{
    /// Resizes the array in-place to the given shape.
    pub fn resize(&mut self, shape: [usize; N], value: T) {
        assert!(self.layout.len() == 0); // TODO: Fix generic resize

        let len = shape.iter().product();

        self.buffer.vec.resize(len, value);
        self.layout.resize(shape);
    }
}

impl<T, B: Buffer<T>, L: Layout<N, 0>, const N: usize> Deref for ArrayBase<T, B, L, N> {
    type Target = ViewBase<T, L, N, 0>;

    fn deref(&self) -> &Self::Target {
        let data = self.buffer.as_ptr().cast();
        let layout = (&self.layout) as *const L as usize;

        unsafe { &*(ptr::from_raw_parts(data, layout) as *const Self::Target) }
    }
}

impl<T, B: Buffer<T>, L: Layout<N, 0>, const N: usize> DerefMut for ArrayBase<T, B, L, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let data = self.buffer.as_mut_ptr().cast();
        let layout = (&mut self.layout) as *mut L as usize;

        unsafe { &mut *(ptr::from_raw_parts_mut(data, layout) as *mut Self::Target) }
    }
}

impl<T, const L: usize> Buffer<T> for StaticBuffer<T, L> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.array.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.array.as_ptr()
    }
}

impl<T, A: Allocator> Buffer<T> for DenseBuffer<T, A> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }
}
