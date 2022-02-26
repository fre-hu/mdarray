use std::alloc::{Allocator, Global};
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;
use std::ops::{Range, RangeBounds};
use std::{mem, ptr, slice};

use crate::dim::{Const, Dim, U1};
use crate::format::Format;
use crate::grid::{DenseGrid, SubGrid, SubGridMut};
use crate::index::{panic_bounds_check, SpanIndex};
use crate::iter::{AxisIter, AxisIterMut};
use crate::layout::{DenseLayout, HasLinearIndexing, HasSliceIndexing, Layout, StaticLayout};
use crate::mapping::Mapping;
use crate::order::Order;

/// Multidimensional array span with static rank and element order.
#[repr(transparent)]
pub struct SpanBase<T, L: Copy> {
    phantom: PhantomData<(T, L)>,
    _slice: [()],
}

impl<T, L: Copy> SpanBase<T, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        (self as *mut Self).cast()
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        (self as *const Self).cast()
    }

    /// Creates an array span from a raw pointer and an array layout.
    /// # Safety
    /// The pointer must be non-null and a valid array span for the given layout.
    pub unsafe fn from_raw_parts(ptr: *const T, layout: &L) -> *const Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        if mem::size_of::<L>() == 0 {
            ptr::from_raw_parts(ptr.cast(), 0usize)
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::from_raw_parts(ptr.cast(), mem::transmute_copy(layout))
        } else {
            ptr::from_raw_parts(ptr.cast(), layout as *const L as usize)
        }
    }

    /// Creates a mutable array span from a raw pointer and an array layout.
    /// # Safety
    /// The pointer must be non-null and a valid array span for the given layout.
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, layout: &L) -> *mut Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        if mem::size_of::<L>() == 0 {
            ptr::from_raw_parts_mut(ptr.cast(), 0usize)
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::from_raw_parts_mut(ptr.cast(), mem::transmute_copy(layout))
        } else {
            ptr::from_raw_parts_mut(ptr.cast(), layout as *const L as usize)
        }
    }

    /// Returns the array layout.
    pub fn layout(&self) -> L {
        let layout = ptr::metadata(self);

        if mem::size_of::<L>() == 0 {
            unsafe { mem::transmute_copy(&()) }
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            unsafe { mem::transmute_copy(&layout) }
        } else {
            unsafe { *(layout as *const L) }
        }
    }
}

impl<T, D: Dim, F: Format, O: Order> SpanBase<T, Layout<D, F, O>> {
    /// Returns a mutable slice of all elements in the array.
    /// # Panics
    /// Panics if the array layout is not contiguous.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "array layout not contiguous");

        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    /// # Panics
    /// Panics if the array layout is not contiguous.
    pub fn as_slice(&self) -> &[T] {
        assert!(self.is_contiguous(), "array layout not contiguous");

        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns an iterator that gives array views over the specified middle dimension.
    ///
    /// Iterating over middle dimensions maintains the unit inner stride propery however not
    /// uniform stride, so that the resulting array views have general or strided format.
    /// # Panics
    /// Panics if the inner or outer dimension is specified, as that would affect the return type.
    pub fn axis_iter(&self, dim: usize) -> AxisIter<T, Layout<D::Lower, F::NonUniform, O>> {
        assert!(dim > 0 && dim + 1 < self.rank(), "inner or outer dimension not allowed");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().to_non_uniform().remove_dim(dim),
                self.size(dim),
                self.stride(dim),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the specified middle dimension.
    ///
    /// Iterating over middle dimensions maintains the unit inner stride propery however not
    /// uniform stride, so that the resulting array views have general or strided format.
    /// # Panics
    /// Panics if the inner or outer dimension is specified, as that would affect the return type.
    pub fn axis_iter_mut(
        &mut self,
        dim: usize,
    ) -> AxisIterMut<T, Layout<D::Lower, F::NonUniform, O>> {
        assert!(dim > 0 && dim + 1 < self.rank(), "inner or outer dimension not allowed");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().to_non_uniform().remove_dim(dim),
                self.size(dim),
                self.stride(dim),
            )
        }
    }

    /// Clones an array span into the array span.
    /// # Panics
    /// Panics if the two spans have different shapes.
    pub fn clone_from_span(&mut self, span: &SpanBase<T, Layout<D, impl Format, O>>)
    where
        T: Clone,
    {
        clone_span(span, self);
    }

    /// Returns the dimension with the specified index, counted from the innermost dimension.
    pub fn dim(&self, index: usize) -> usize {
        self.layout().dim(index)
    }

    /// Returns the dimensions with the specified indicies, counted from the innermost dimension.
    pub fn dims(&self, indices: impl RangeBounds<usize>) -> Range<usize> {
        self.layout().dims(indices)
    }

    /// Fills the array span with elements by cloning `value`.
    pub fn fill(&mut self, value: impl Borrow<T> + Copy)
    where
        T: Clone,
    {
        fill_with(self, &mut || value.borrow().clone());
    }

    /// Fills the array span with elements returned by calling a closure repeatedly.
    pub fn fill_with(&mut self, mut f: impl FnMut() -> T) {
        fill_with(self, &mut f);
    }

    /// Returns an iterator over the flattened array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    pub fn flat_iter(&self) -> F::Iter<'_, T> {
        F::Mapping::iter(self)
    }

    /// Returns a mutable iterator over the flattened array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    pub fn flat_iter_mut(&mut self) -> F::IterMut<'_, T> {
        F::Mapping::iter_mut(self)
    }

    /// Copies the specified subarray into a new array.
    pub fn grid<I: SpanIndex<D, O>>(&self, index: I) -> DenseGrid<T, I::Dim, O>
    where
        T: Clone,
    {
        self.view(index).to_grid()
    }

    /// Copies the specified subarray into a new array with the specified allocator.
    pub fn grid_in<I: SpanIndex<D, O>, A>(&self, index: I, alloc: A) -> DenseGrid<T, I::Dim, O, A>
    where
        T: Clone,
        A: Allocator,
    {
        self.view(index).to_grid_in(alloc)
    }

    /// Returns true if the array layout type supports linear indexing.
    pub fn has_linear_indexing(&self) -> bool {
        self.layout().has_linear_indexing()
    }

    /// Returns true if the array layout type supports slice indexing.
    pub fn has_slice_indexing(&self) -> bool {
        self.layout().has_slice_indexing()
    }

    /// Returns an iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property however not
    /// unit inner stride, so that the resulting array views have linear or strided format.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn inner_iter(&self) -> AxisIter<T, Layout<D::Lower, F::NonUnitStrided, O>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().to_non_unit_strided().remove_dim(self.dim(0)),
                self.size(self.dim(0)),
                self.stride(self.dim(0)),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property however not
    /// unit inner stride, so that the resulting array views have linear or strided format.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn inner_iter_mut(&mut self) -> AxisIterMut<T, Layout<D::Lower, F::NonUnitStrided, O>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().to_non_unit_strided().remove_dim(self.dim(0)),
                self.size(self.dim(0)),
                self.stride(self.dim(0)),
            )
        }
    }

    /// Returns true if the array has column-major element order.
    pub fn is_column_major(&self) -> bool {
        self.layout().is_column_major()
    }

    /// Returns true if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    /// Returns true if the array has row-major element order.
    pub fn is_row_major(&self) -> bool {
        self.layout().is_row_major()
    }

    /// Returns true if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(&self) -> bool {
        self.layout().is_uniformly_strided()
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns an iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same format.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn outer_iter(&self) -> AxisIter<T, Layout<D::Lower, F, O>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        let dim = self.dim(self.rank() - 1);

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().remove_dim(dim),
                self.size(dim),
                self.stride(dim),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same format.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, Layout<D::Lower, F, O>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        let dim = self.dim(self.rank() - 1);

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(dim),
                self.size(dim),
                self.stride(dim),
            )
        }
    }

    /// Returns the rank of the array.
    pub fn rank(&self) -> usize {
        self.layout().rank()
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> D::Shape {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
    }

    /// Divides an array span into two at the specified point along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at(
        &self,
        mid: usize,
    ) -> (SubGrid<T, Layout<D, F, O>>, SubGrid<T, Layout<D, F, O>>) {
        assert!(self.rank() > 0, "invalid rank");

        let dim = self.dim(self.rank() - 1);

        if mid > self.size(dim) {
            panic_bounds_check(mid, self.size(dim));
        }

        let first_layout = self.layout().resize_dim(dim, mid);
        let second_layout = self.layout().resize_dim(dim, self.size(dim) - mid);

        // Discard invalid offset if the second span is empty.
        let count = if mid == self.size(dim) { 0 } else { self.stride(dim) * mid as isize };

        unsafe {
            (
                SubGrid::new_unchecked(self.as_ptr(), first_layout),
                SubGrid::new_unchecked(self.as_ptr().offset(count), second_layout),
            )
        }
    }

    /// Divides a mutable array span into two at the specified point along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (SubGridMut<T, Layout<D, F, O>>, SubGridMut<T, Layout<D, F, O>>) {
        assert!(self.rank() > 0, "invalid rank");

        let dim = self.dim(self.rank() - 1);

        if mid > self.size(dim) {
            panic_bounds_check(mid, self.size(dim));
        }

        let first_layout = self.layout().resize_dim(dim, mid);
        let second_layout = self.layout().resize_dim(dim, self.size(dim) - mid);

        // Discard invalid offset if the second span is empty.
        let count = if mid == self.size(dim) { 0 } else { self.stride(dim) * mid as isize };

        unsafe {
            (
                SubGridMut::new_unchecked(self.as_mut_ptr(), first_layout),
                SubGridMut::new_unchecked(self.as_mut_ptr().offset(count), second_layout),
            )
        }
    }

    /// Returns the distance between elements in the specified dimension.
    pub fn stride(&self, dim: usize) -> isize {
        self.layout().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> D::Strides {
        self.layout().strides()
    }

    /// Copies the array span into a new array.
    pub fn to_grid(&self) -> DenseGrid<T, D, O>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> DenseGrid<T, D, O, A>
    where
        T: Clone,
    {
        let mut grid = DenseGrid::new_in(alloc);

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_vec_in(Global)
    }

    /// Copies the array span into a new vector with the specified allocator.
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_grid_in(alloc).into_vec()
    }

    /// Returns an array view for the specified subarray.
    pub fn view<I>(&self, index: I) -> SubGrid<T, Layout<I::Dim, I::Format<F>, O>>
    where
        I: SpanIndex<D, O>,
    {
        let (offset, layout, _) = I::span_info(index, self.layout());
        let count = if layout.is_empty() { 0 } else { offset }; // Discard offset if empty view.

        unsafe { SubGrid::new_unchecked(self.as_ptr().offset(count), layout) }
    }

    /// Returns a mutable array view for the specified subarray.
    pub fn view_mut<I>(&mut self, index: I) -> SubGridMut<T, Layout<I::Dim, I::Format<F>, O>>
    where
        I: SpanIndex<D, O>,
    {
        let (offset, layout, _) = I::span_info(index, self.layout());
        let count = if layout.is_empty() { 0 } else { offset }; // Discard offset if empty view.

        unsafe { SubGridMut::new_unchecked(self.as_mut_ptr().offset(count), layout) }
    }

    /// Returns an array view of the entire array span.
    pub fn to_view(&self) -> SubGrid<T, Layout<D, F, O>> {
        unsafe { SubGrid::new_unchecked(self.as_ptr(), self.layout()) }
    }

    /// Returns a mutable array view of the entire array span.
    pub fn to_view_mut(&mut self) -> SubGridMut<T, Layout<D, F, O>> {
        unsafe { SubGridMut::new_unchecked(self.as_mut_ptr(), self.layout()) }
    }
}

impl<T, D: Dim, F: Format, O: Order> SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasLinearIndexing,
{
    /// Returns an iterator over the array span, which must support linear indexing.
    pub fn iter(&self) -> F::Iter<'_, T> {
        F::Mapping::iter(self)
    }

    /// Returns a mutable iterator over the array span, which must support linear indexing.
    pub fn iter_mut(&mut self) -> F::IterMut<'_, T> {
        F::Mapping::iter_mut(self)
    }
}

impl<T, D: Dim, F: Format, O: Order> AsMut<[T]> for SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasSliceIndexing,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, D: Dim, F: Format, O: Order> AsRef<[T]> for SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasSliceIndexing,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, O: Order> AsMut<SpanBase<T, DenseLayout<U1, O>>> for [T] {
    fn as_mut(&mut self) -> &mut SpanBase<T, DenseLayout<U1, O>> {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        unsafe { &mut *ptr::from_raw_parts_mut(self.as_mut_ptr().cast(), self.len()) }
    }
}

impl<T, O: Order> AsRef<SpanBase<T, DenseLayout<U1, O>>> for [T] {
    fn as_ref(&self) -> &SpanBase<T, DenseLayout<U1, O>> {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        unsafe { &*ptr::from_raw_parts(self.as_ptr().cast(), self.len()) }
    }
}

macro_rules! impl_as_mut_ref_array {
    ($n:tt, ($($xyz:tt),+), ($($zyx:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> AsMut<SpanBase<T, DenseLayout<Const<$n>, O>>>
            for $array
        {
            fn as_mut(&mut self) -> &mut SpanBase<T, DenseLayout<Const<$n>, O>> {
                let layout = O::select(
                    &<($(Const<$xyz>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                    &<($(Const<$zyx>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                );

                unsafe { &mut *SpanBase::from_raw_parts_mut(self.as_mut_ptr().cast(), layout) }
            }
        }

        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> AsRef<SpanBase<T, DenseLayout<Const<$n>, O>>>
            for $array
        {
            fn as_ref(&self) -> &SpanBase<T, DenseLayout<Const<$n>, O>> {
                let layout = O::select(
                    &<($(Const<$xyz>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                    &<($(Const<$zyx>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                );

                unsafe { &*SpanBase::from_raw_parts(self.as_ptr().cast(), layout) }
            }
        }
    };
}

impl_as_mut_ref_array!(1, (X), (X), [T; X]);
impl_as_mut_ref_array!(2, (X, Y), (Y, X), [[T; X]; Y]);
impl_as_mut_ref_array!(3, (X, Y, Z), (Z, Y, X), [[[T; X]; Y]; Z]);
impl_as_mut_ref_array!(4, (X, Y, Z, W), (W, Z, Y, X), [[[[T; X]; Y]; Z]; W]);
impl_as_mut_ref_array!(5, (X, Y, Z, W, U), (U, W, Z, Y, X), [[[[[T; X]; Y]; Z]; W]; U]);
impl_as_mut_ref_array!(6, (X, Y, Z, W, U, V), (V, U, W, Z, Y, X), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T: Debug, D: Dim, F: Format, O: Order> Debug for SpanBase<T, Layout<D, F, O>> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        if self.rank() == 0 {
            self[D::Shape::default()].fmt(fmt)
        } else if self.rank() == 1 {
            fmt.debug_list().entries(self.flat_iter()).finish()
        } else {
            fmt.debug_list().entries(self.outer_iter()).finish()
        }
    }
}

impl<'a, T, D: Dim, F: Format, O: Order> IntoIterator for &'a SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasLinearIndexing,
{
    type Item = &'a T;
    type IntoIter = F::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: Dim, F: Format, O: Order> IntoIterator for &'a mut SpanBase<T, Layout<D, F, O>>
where
    Layout<D, F, O>: HasLinearIndexing,
{
    type Item = &'a mut T;
    type IntoIter = F::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Clone, D: Dim, O: Order> ToOwned for SpanBase<T, DenseLayout<D, O>> {
    type Owned = DenseGrid<T, D, O>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }
}

fn clone_span<T: Clone, D: Dim, O: Order>(
    src: &SpanBase<T, Layout<D, impl Format, O>>,
    dst: &mut SpanBase<T, Layout<D, impl Format, O>>,
) {
    if src.has_linear_indexing() && dst.has_linear_indexing() {
        assert!(src.shape().as_ref() == dst.shape().as_ref(), "shape mismatch");

        if src.has_slice_indexing() && dst.has_slice_indexing() {
            dst.as_mut_slice().clone_from_slice(src.as_slice());
        } else {
            for (x, y) in dst.flat_iter_mut().zip(src.flat_iter()) {
                x.clone_from(y);
            }
        }
    } else {
        let dim = src.dim(src.rank() - 1);

        assert!(src.size(dim) == dst.size(dim), "shape mismatch");

        for (mut x, y) in dst.outer_iter_mut().zip(src.outer_iter()) {
            clone_span(&y, &mut x);
        }
    }
}

fn fill_with<T>(
    span: &mut SpanBase<T, Layout<impl Dim, impl Format, impl Order>>,
    f: &mut impl FnMut() -> T,
) {
    if span.has_linear_indexing() {
        for x in span.flat_iter_mut() {
            *x = f();
        }
    } else {
        for mut x in span.outer_iter_mut() {
            fill_with(&mut x, f);
        }
    }
}