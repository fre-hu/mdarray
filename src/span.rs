#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::{mem, ptr, slice};

use crate::dim::{Dim, Rank, Shape};
use crate::format::{Format, Uniform};
use crate::grid::{DenseGrid, SubGrid, SubGridMut};
use crate::index::{Axis, Const, Params, SpanIndex, ViewIndex};
use crate::iter::{AxisIter, AxisIterMut};
use crate::layout::{DenseLayout, Layout, ValidLayout, ViewLayout};
use crate::mapping::Mapping;

/// Multidimensional array span with static rank and element order.
#[repr(transparent)]
pub struct SpanBase<T, L: Copy> {
    phantom: PhantomData<(T, L)>,
    slice: [()],
}

pub type DenseSpan<T, D> = SpanBase<T, DenseLayout<D>>;

impl<T, L: Copy> SpanBase<T, L> {
    /// Returns a mutable pointer to the array buffer.
    #[cfg(not(feature = "permissive-provenance"))]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        (self as *mut Self).cast()
    }

    /// Returns a mutable pointer to the array buffer.
    #[cfg(feature = "permissive-provenance")]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut Self as *mut () as usize as *mut T // Use previously exposed provenance.
    }

    /// Returns a raw pointer to the array buffer.
    #[cfg(not(feature = "permissive-provenance"))]
    pub fn as_ptr(&self) -> *const T {
        (self as *const Self).cast()
    }

    /// Returns a raw pointer to the array buffer.
    #[cfg(feature = "permissive-provenance")]
    pub fn as_ptr(&self) -> *const T {
        self as *const Self as *const () as usize as *const T // Use previously exposed provenance.
    }

    /// Returns the array layout.
    pub fn layout(&self) -> L {
        let len = self.slice.len();

        if mem::size_of::<L>() == 0 {
            unsafe { mem::transmute_copy(&()) }
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            unsafe { mem::transmute_copy(&len) }
        } else {
            unsafe { *(len as *const L) }
        }
    }

    /// Returns an array view of the entire array span.
    pub fn to_view(&self) -> SubGrid<T, L> {
        unsafe { SubGrid::new_unchecked(self.as_ptr(), self.layout()) }
    }

    /// Returns a mutable array view of the entire array span.
    pub fn to_view_mut(&mut self) -> SubGridMut<T, L> {
        unsafe { SubGridMut::new_unchecked(self.as_mut_ptr(), self.layout()) }
    }

    pub(crate) unsafe fn from_raw_parts(ptr: *const T, layout: &L) -> *const Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        #[cfg(not(feature = "permissive-provenance"))]
        let ptr = ptr as *const (); // Assume that provenance is maintained for &[()].
        #[cfg(feature = "permissive-provenance")]
        let ptr = ptr as usize as *const (); // Expose pointer provenance.

        if mem::size_of::<L>() == 0 {
            ptr::slice_from_raw_parts(ptr, 0) as *const Self
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::slice_from_raw_parts(ptr, mem::transmute_copy(layout)) as *const Self
        } else {
            ptr::slice_from_raw_parts(ptr, layout as *const L as usize) as *const Self
        }
    }

    pub(crate) unsafe fn from_raw_parts_mut(ptr: *mut T, layout: &L) -> *mut Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        #[cfg(not(feature = "permissive-provenance"))]
        let ptr = ptr as *mut (); // Assume that provenance is maintained for &mut [()].
        #[cfg(feature = "permissive-provenance")]
        let ptr = ptr as usize as *mut (); // Expose pointer provenance.

        if mem::size_of::<L>() == 0 {
            ptr::slice_from_raw_parts_mut(ptr, 0) as *mut Self
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::slice_from_raw_parts_mut(ptr, mem::transmute_copy(layout)) as *mut Self
        } else {
            ptr::slice_from_raw_parts_mut(ptr, layout as *const L as usize) as *mut Self
        }
    }
}

impl<T, D: Dim, F: Format> SpanBase<T, Layout<D, F>> {
    /// Returns an iterator that gives array views over the specified dimension.
    ///
    /// When iterating over the outer dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same format.
    ///
    /// When iterating over the inner dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided format.
    ///
    /// When iterating over the middle dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided format.
    pub fn axis_iter<const DIM: usize>(
        &self,
    ) -> AxisIter<T, Layout<D::Lower, <Const<DIM> as Axis<D>>::Remove<F>>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().remove_dim(DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the specified dimension.
    ///
    /// When iterating over the outer dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same format.
    ///
    /// When iterating over the inner dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided format.
    ///
    /// When iterating over the middle dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided format.
    pub fn axis_iter_mut<const DIM: usize>(
        &mut self,
    ) -> AxisIterMut<T, Layout<D::Lower, <Const<DIM> as Axis<D>>::Remove<F>>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
    }

    /// Clones an array span into the array span.
    /// # Panics
    /// Panics if the two spans have different shapes.
    pub fn clone_from_span(&mut self, span: &SpanBase<T, Layout<D, impl Format>>)
    where
        T: Clone,
    {
        clone_span(span, self);
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

    /// Returns a one-dimensional array view of the array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten(&self) -> SubGrid<T, Layout<Rank<1, D::Order>, F::Uniform>> {
        self.to_view().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten_mut(&mut self) -> SubGridMut<T, Layout<Rank<1, D::Order>, F::Uniform>> {
        self.to_view_mut().into_flattened()
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    /// # Safety
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: SpanIndex<T, Layout<D, F>>,
    {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    /// # Safety
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: SpanIndex<T, Layout<D, F>>,
    {
        index.get_unchecked_mut(self)
    }

    /// Copies the specified subarray into a new array.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    pub fn grid<P: Params, I>(&self, index: I) -> DenseGrid<T, P::Dim>
    where
        T: Clone,
        I: ViewIndex<D, F, Params = P>,
    {
        self.view(index).to_grid()
    }

    /// Copies the specified subarray into a new array with the specified allocator.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[cfg(feature = "nightly")]
    pub fn grid_in<P: Params, I, A>(&self, index: I, alloc: A) -> DenseGrid<T, P::Dim, A>
    where
        T: Clone,
        I: ViewIndex<D, F, Params = P>,
        A: Allocator,
    {
        self.view(index).to_grid_in(alloc)
    }

    /// Returns an iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property but not
    /// unit inner stride, so that the resulting array views have flat or strided format.
    /// # Panics
    /// Panics if the rank is not at least 1.
    pub fn inner_iter(&self) -> AxisIter<T, ValidLayout<D::Lower, F::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().remove_dim(D::dim(0)),
                self.size(D::dim(0)),
                self.stride(D::dim(0)),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property but not
    /// unit inner stride, so that the resulting array views have flat or strided format.
    /// # Panics
    /// Panics if the rank is not at least 1.
    pub fn inner_iter_mut(&mut self) -> AxisIterMut<T, ValidLayout<D::Lower, F::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(D::dim(0)),
                self.size(D::dim(0)),
                self.stride(D::dim(0)),
            )
        }
    }

    /// Returns true if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    /// Returns true if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(&self) -> bool {
        self.layout().is_uniformly_strided()
    }

    /// Returns an iterator over the array span, which must support linear indexing.
    pub fn iter(&self) -> F::Iter<'_, T>
    where
        F: Uniform,
    {
        F::Mapping::iter(self)
    }

    /// Returns a mutable iterator over the array span, which must support linear indexing.
    pub fn iter_mut(&mut self) -> F::IterMut<'_, T>
    where
        F: Uniform,
    {
        F::Mapping::iter_mut(self)
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
    /// Panics if the rank is not at least 1.
    pub fn outer_iter(&self) -> AxisIter<T, ValidLayout<D::Lower, F>> {
        assert!(D::RANK > 0, "invalid rank");

        let dim = D::dim(D::RANK - 1);

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
    /// Panics if the rank is not at least 1.
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, ValidLayout<D::Lower, F>> {
        assert!(D::RANK > 0, "invalid rank");

        let dim = D::dim(D::RANK - 1);

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(dim),
                self.size(dim),
                self.stride(dim),
            )
        }
    }

    /// Returns a reformatted array view of the array span.
    /// # Panics
    /// Panics if the array layout is not compatible with the new format.
    pub fn reformat<G: Format>(&self) -> SubGrid<T, Layout<D, G>> {
        self.to_view().into_format()
    }

    /// Returns a mutable reformatted array view of the array span.
    /// # Panics
    /// Panics if the array layout is not compatible with the new format.
    pub fn reformat_mut<G: Format>(&mut self) -> SubGridMut<T, Layout<D, G>> {
        self.to_view_mut().into_format()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape<S: Shape>(&self, shape: S) -> SubGrid<T, ValidLayout<S::Dim<D::Order>, F>> {
        self.to_view().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape_mut<S: Shape>(
        &mut self,
        shape: S,
    ) -> SubGridMut<T, ValidLayout<S::Dim<D::Order>, F>> {
        self.to_view_mut().into_shape(shape)
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> D::Shape {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
    }

    /// Divides an array span into two at an index along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at(&self, mid: usize) -> (SubGrid<T, Layout<D, F>>, SubGrid<T, Layout<D, F>>) {
        self.to_view().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (SubGridMut<T, Layout<D, F>>, SubGridMut<T, Layout<D, F>>) {
        self.to_view_mut().into_split_at(mid)
    }

    /// Divides an array span into two at an index along the specified dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at<const DIM: usize>(
        &self,
        mid: usize,
    ) -> (
        SubGrid<T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>,
        SubGrid<T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view().into_split_axis_at(mid)
    }

    /// Divides a mutable array span into two at an index along the specified dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at_mut<const DIM: usize>(
        &mut self,
        mid: usize,
    ) -> (
        SubGridMut<T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>,
        SubGridMut<T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view_mut().into_split_axis_at(mid)
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
    #[cfg(not(feature = "nightly"))]
    pub fn to_grid(&self) -> DenseGrid<T, D>
    where
        T: Clone,
    {
        let mut grid = DenseGrid::new();

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new array.
    #[cfg(feature = "nightly")]
    pub fn to_grid(&self) -> DenseGrid<T, D>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> DenseGrid<T, D, A>
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
        self.to_grid().into_vec()
    }

    /// Copies the array span into a new vector with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_grid_in(alloc).into_vec()
    }

    /// Returns an array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    pub fn view<I>(&self, index: I) -> SubGrid<T, ViewLayout<I::Params>>
    where
        I: ViewIndex<D, F>,
    {
        self.to_view().into_view(index)
    }

    /// Returns a mutable array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    pub fn view_mut<I>(&mut self, index: I) -> SubGridMut<T, ViewLayout<I::Params>>
    where
        I: ViewIndex<D, F>,
    {
        self.to_view_mut().into_view(index)
    }
}

impl<T, D: Dim> DenseSpan<T, D> {
    /// Returns a mutable slice of all elements in the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, D: Dim> AsMut<[T]> for DenseSpan<T, D> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, D: Dim> AsRef<[T]> for DenseSpan<T, D> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Debug, D: Dim, F: Format> Debug for SpanBase<T, Layout<D, F>> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        if D::RANK == 0 {
            self[D::Shape::default()].fmt(fmt)
        } else if D::RANK == 1 {
            fmt.debug_list().entries(self.flatten().iter()).finish()
        } else {
            fmt.debug_list().entries(self.outer_iter()).finish()
        }
    }
}

impl<T, L: Copy, I: SpanIndex<T, L>> Index<I> for SpanBase<T, L> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, L: Copy, I: SpanIndex<T, L>> IndexMut<I> for SpanBase<T, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, D: Dim, F: Uniform> IntoIterator for &'a SpanBase<T, Layout<D, F>> {
    type Item = &'a T;
    type IntoIter = F::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: Dim, F: Uniform> IntoIterator for &'a mut SpanBase<T, Layout<D, F>> {
    type Item = &'a mut T;
    type IntoIter = F::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Clone, D: Dim> ToOwned for DenseSpan<T, D> {
    type Owned = DenseGrid<T, D>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }
}

fn clone_span<T: Clone, D: Dim, F: Format, G: Format>(
    src: &SpanBase<T, Layout<D, F>>,
    dst: &mut SpanBase<T, Layout<D, G>>,
) {
    if F::IS_UNIFORM && G::IS_UNIFORM {
        assert!(src.shape()[..] == dst.shape()[..], "shape mismatch");

        if F::IS_UNIT_STRIDED && G::IS_UNIT_STRIDED {
            dst.reformat_mut().as_mut_slice().clone_from_slice(src.reformat().as_slice());
        } else {
            for (x, y) in dst.flatten_mut().iter_mut().zip(src.flatten().iter()) {
                x.clone_from(y);
            }
        }
    } else {
        let dim = D::dim(D::RANK - 1);

        assert!(src.size(dim) == dst.size(dim), "shape mismatch");

        for (mut x, y) in dst.outer_iter_mut().zip(src.outer_iter()) {
            clone_span(&y, &mut x);
        }
    }
}

fn fill_with<T, F: Format>(span: &mut SpanBase<T, Layout<impl Dim, F>>, f: &mut impl FnMut() -> T) {
    if F::IS_UNIFORM {
        for x in span.flatten_mut().iter_mut() {
            *x = f();
        }
    } else {
        for mut x in span.outer_iter_mut() {
            fill_with(&mut x, f);
        }
    }
}
