use std::alloc::{Allocator, Global};
use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Formatter, Result};
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};
use std::{cmp, mem};

use crate::buffer::{Buffer, BufferMut, DenseBuffer, SubBuffer, SubBufferMut};
use crate::dimension::{Const, Dim, Shape};
use crate::layout::{DenseLayout, Layout, StaticLayout};
use crate::mapping::Mapping;
use crate::order::Order;
use crate::span::SpanBase;

/// Multidimensional array with static rank and element order.
#[derive(Clone)]
pub struct GridBase<B: Buffer> {
    buffer: B,
}

/// Dense multidimensional array with static rank and element order.
pub type DenseGrid<T, D, O, A = Global> = GridBase<DenseBuffer<T, D, O, A>>;

/// Multidimensional array view with static rank and element order.
pub type SubGrid<'a, T, L> = GridBase<SubBuffer<'a, T, L>>;

/// Mutable multidimensional array view with static rank and element order.
pub type SubGridMut<'a, T, L> = GridBase<SubBufferMut<'a, T, L>>;

impl<B: Buffer> GridBase<B> {
    /// Returns a mutable array span of the entire array.
    pub fn as_mut_span(&mut self) -> &mut SpanBase<B::Item, B::Layout>
    where
        B: BufferMut,
    {
        unsafe {
            &mut *SpanBase::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.layout())
        }
    }

    /// Returns an array span of the entire array.
    pub fn as_span(&self) -> &SpanBase<B::Item, B::Layout> {
        unsafe { &*SpanBase::from_raw_parts(self.buffer.as_ptr(), self.buffer.layout()) }
    }
}

impl<T, D: Dim, O: Order, A: Allocator> DenseGrid<T, D, O, A> {
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
    /// # Safety
    /// The pointer must be a valid allocation given the shape, capacity and allocator.
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: D::Shape,
        capacity: usize,
        alloc: A,
    ) -> Self {
        Self { buffer: DenseBuffer::from_raw_parts_in(ptr, shape, capacity, alloc) }
    }

    /// Decomposes an array into its raw components including the allocator.
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, D::Shape, usize, A) {
        let Self { buffer, .. } = self;

        buffer.into_raw_parts_with_alloc()
    }

    /// Converts the array into a one-dimensional array, which must have the same length.
    /// # Panics
    /// Panics if the array length is changed.
    pub fn flatten(self) -> DenseGrid<T, Const<1>, O, A> {
        let len = self.len();

        self.reshape([len])
    }

    /// Converts the array into a vector.
    pub fn into_vec(self) -> Vec<T, A> {
        let len = self.len();
        let (ptr, _, capacity, alloc) = self.into_raw_parts_with_alloc();

        unsafe { Vec::from_raw_parts_in(ptr, len, capacity, alloc) }
    }

    /// Creates a new, empty array with the specified allocator.
    pub fn new_in(alloc: A) -> Self {
        Self { buffer: DenseBuffer::new_in(alloc) }
    }

    /// Converts the array into a reshaped array, which must have the same length.
    /// # Panics
    /// Panics if the array length is changed.
    pub fn reshape<S: Shape>(self, shape: S) -> DenseGrid<T, S::Dim, O, A> {
        let layout = self.layout().reshape(shape);
        let (ptr, _, capacity, alloc) = self.into_raw_parts_with_alloc();

        unsafe { DenseGrid::from_raw_parts_in(ptr, layout.shape(), capacity, alloc) }
    }

    /// Resizes the array to the given shape.
    pub fn resize(&mut self, shape: D::Shape, value: T)
    where
        T: Clone,
        A: Clone,
    {
        self.buffer.resize(shape, value);
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
        Self { buffer: DenseBuffer::with_capacity_in(capacity, alloc) }
    }
}

impl<T, D: Dim, O: Order> DenseGrid<T, D, O, Global> {
    /// Creates an array from raw components of another array.
    /// # Safety
    /// The pointer must be a valid allocation given the shape and capacity.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: D::Shape, capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, shape, capacity, Global)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, D::Shape, usize) {
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

macro_rules! impl_sub_grid {
    ($type:tt, $buffer:tt, $as_ptr:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T, L: Layout> $type<'a, T, L> {
            /// Converts the array view into a one-dimensional array view.
            /// # Panics
            /// Panics if the array layout is not compatible with linear indexing and fixed stride.
            pub fn flatten(self) -> $type<'a, T, L::Reshaped<[usize; 1]>> {
                let len = self.len();

                self.reshape([len])
            }

            /// Converts the array view into a dense array view.
            /// # Panics
            /// Panics if the array layout is not contiguous.
            pub fn into_dense($($mut)? self) -> $type<'a, T, DenseLayout<L::Dim, L::Order>> {
                unsafe { $type::new(self.$as_ptr(), self.layout().to_dense()) }
            }

            /// Converts the array view into a dense or general array view.
            /// # Panics
            /// Panics if the innermost stride is not unitary.
            pub fn into_unit_strided($($mut)? self) -> $type<'a, T, L::UnitStrided> {
                unsafe { $type::new(self.$as_ptr(), self.layout().to_unit_strided()) }
            }

            /// Creates an array view from a raw pointer and layout.
            /// # Safety
            /// The pointer must be a valid array view for the given layout.
            pub unsafe fn new(ptr: *$raw_mut T, layout: L) -> Self {
                Self { buffer: $buffer::new(ptr, layout) }
            }

            /// Converts the array view into a reshaped array view with compatible layout.
            /// # Panics
            /// Panics if the array length is changed, or the memory layout is not compatible.
            pub fn reshape<S: Shape>($($mut)? self, shape: S) -> $type<'a, T, L::Reshaped<S>> {
                unsafe { $type::new(self.$as_ptr(), self.layout().reshape(shape)) }
            }
        }
    };
}

impl_sub_grid!(SubGrid, SubBuffer, as_ptr, const, {});
impl_sub_grid!(SubGridMut, SubBufferMut, as_mut_ptr, mut, {mut});

impl<T, D: Dim, O: Order, A: Allocator> Borrow<SpanBase<T, DenseLayout<D, O>>>
    for DenseGrid<T, D, O, A>
{
    fn borrow(&self) -> &SpanBase<T, DenseLayout<D, O>> {
        self.as_span()
    }
}

impl<T, D: Dim, O: Order, A: Allocator> BorrowMut<SpanBase<T, DenseLayout<D, O>>>
    for DenseGrid<T, D, O, A>
{
    fn borrow_mut(&mut self) -> &mut SpanBase<T, DenseLayout<D, O>> {
        self.as_mut_span()
    }
}

impl<T: Debug, B: Buffer<Item = T>> Debug for GridBase<B> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        self.as_span().fmt(fmt)
    }
}

impl<T, D: Dim, O: Order> Default for DenseGrid<T, D, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Buffer> Deref for GridBase<B> {
    type Target = SpanBase<B::Item, B::Layout>;

    fn deref(&self) -> &Self::Target {
        self.as_span()
    }
}

impl<B: BufferMut> DerefMut for GridBase<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_span()
    }
}

impl<T, O: Order, A: Allocator + Clone> Extend<T> for DenseGrid<T, Const<1>, O, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut grid = Self::new_in(self.allocator().clone());

        mem::swap(self, &mut grid);

        let mut vec = grid.into_vec();

        vec.extend(iter);

        mem::swap(self, &mut Self::from(vec));
    }
}

impl<T: Clone, O: Order> From<&[T]> for DenseGrid<T, Const<1>, O> {
    fn from(slice: &[T]) -> Self {
        Self::from(slice.to_vec())
    }
}

macro_rules! impl_from_array {
    ($n:tt, ($($xyz:tt),+), ($($zyx:tt),+), $array:tt) => {
        #[allow(clippy::type_complexity)]
        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> From<$array> for DenseGrid<T, Const<$n>, O> {
            fn from(array: $array) -> Self {
                let (ptr, _, mut capacity, alloc) = Vec::from(array).into_raw_parts_with_alloc();

                let layout = O::select(
                    &<($(Const<$xyz>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                    &<($(Const<$zyx>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                );

                capacity *= unsafe { (mem::size_of_val(&*ptr) / mem::size_of::<T>()) };

                unsafe {
                    Self::from_raw_parts_in(ptr as *mut T, layout.shape(), capacity, alloc)
                }
            }
        }
    };
}

impl_from_array!(1, (X), (X), [T; X]);
impl_from_array!(2, (X, Y), (Y, X), [[T; X]; Y]);
impl_from_array!(3, (X, Y, Z), (Z, Y, X), [[[T; X]; Y]; Z]);
impl_from_array!(4, (X, Y, Z, W), (W, Z, Y, X), [[[[T; X]; Y]; Z]; W]);
impl_from_array!(5, (X, Y, Z, W, U), (U, W, Z, Y, X), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array!(6, (X, Y, Z, W, U, V), (V, U, W, Z, Y, X), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T, O: Order, A: Allocator> From<Vec<T, A>> for DenseGrid<T, Const<1>, O, A> {
    fn from(vec: Vec<T, A>) -> Self {
        let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_alloc();

        unsafe { Self::from_raw_parts_in(ptr, [len], capacity, alloc) }
    }
}

impl<T, O: Order, A: Allocator> From<DenseGrid<T, Const<1>, O, A>> for Vec<T, A> {
    fn from(grid: DenseGrid<T, Const<1>, O, A>) -> Self {
        grid.into_vec()
    }
}

impl<T, O: Order> FromIterator<T> for DenseGrid<T, Const<1>, O> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<'a, B: Buffer> IntoIterator for &'a GridBase<B>
where
    &'a SpanBase<B::Item, B::Layout>: IntoIterator<Item = &'a B::Item>,
{
    type Item = &'a B::Item;
    type IntoIter = <&'a SpanBase<B::Item, B::Layout> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_span().into_iter()
    }
}

impl<'a, B: BufferMut> IntoIterator for &'a mut GridBase<B>
where
    &'a mut SpanBase<B::Item, B::Layout>: IntoIterator<Item = &'a mut B::Item>,
{
    type Item = &'a mut B::Item;
    type IntoIter = <&'a mut SpanBase<B::Item, B::Layout> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_span().into_iter()
    }
}

impl<T, O: Order, A: Allocator> IntoIterator for DenseGrid<T, Const<1>, O, A> {
    type Item = T;
    type IntoIter = <Vec<T, A> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}
