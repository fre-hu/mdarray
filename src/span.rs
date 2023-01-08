#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::borrow::Borrow;
use std::slice;

use crate::array::{GridArray, SpanArray, ViewArray, ViewArrayMut};
use crate::dim::{Const, Dim, Shape};
use crate::format::{Dense, Format, Uniform};
use crate::index::axis::Axis;
use crate::index::span::SpanIndex;
use crate::index::view::{Params, ViewIndex};
use crate::iter::sources::{AxisIter, AxisIterMut};
use crate::layout::Layout;
use crate::mapping::Mapping;
use crate::raw_span::RawSpan;

impl<T, D: Dim, F: Format> SpanArray<T, D, F> {
    /// Returns a mutable pointer to the array buffer.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        RawSpan::from_mut_buffer(&mut self.buffer).as_mut_ptr()
    }

    /// Returns a raw pointer to the array buffer.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        RawSpan::from_buffer(&self.buffer).as_ptr()
    }

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
    ) -> AxisIter<T, D::Lower, <Const<DIM> as Axis<D>>::Remove<F>>
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
    ) -> AxisIterMut<T, D::Lower, <Const<DIM> as Axis<D>>::Remove<F>>
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
    pub fn clone_from_span(&mut self, src: &SpanArray<T, D, impl Format>)
    where
        T: Clone,
    {
        clone_from_span(self, src);
    }

    /// Fills the array span with elements by cloning `value`.
    pub fn fill(&mut self, value: impl Borrow<T>)
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
    #[must_use]
    pub fn flatten(&self) -> ViewArray<T, Const<1>, F::Uniform> {
        self.to_view().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    #[must_use]
    pub fn flatten_mut(&mut self) -> ViewArrayMut<T, Const<1>, F::Uniform> {
        self.to_view_mut().into_flattened()
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    /// # Safety
    /// The index must be within bounds of the array span.
    #[must_use]
    pub unsafe fn get_unchecked<I: SpanIndex<T, D, F>>(&self, index: I) -> &I::Output {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    /// # Safety
    /// The index must be within bounds of the array span.
    #[must_use]
    pub unsafe fn get_unchecked_mut<I: SpanIndex<T, D, F>>(&mut self, index: I) -> &mut I::Output {
        index.get_unchecked_mut(self)
    }

    /// Copies the specified subarray into a new array.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[must_use]
    pub fn grid<P: Params, I>(&self, index: I) -> GridArray<T, P::Dim>
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
    #[must_use]
    pub fn grid_in<P: Params, I, A>(&self, index: I, alloc: A) -> GridArray<T, P::Dim, A>
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
    pub fn inner_iter(
        &self,
    ) -> AxisIter<T, D::Lower, <D::Lower as Dim>::Format<F::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().remove_dim(0),
                self.size(0),
                self.stride(0),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property but not
    /// unit inner stride, so that the resulting array views have flat or strided format.
    /// # Panics
    /// Panics if the rank is not at least 1.
    pub fn inner_iter_mut(
        &mut self,
    ) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Format<F::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(0),
                self.size(0),
                self.stride(0),
            )
        }
    }

    /// Returns true if the array strides are consistent with contiguous memory layout.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns true if the array contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    /// Returns true if the array strides are consistent with uniformly strided memory layout.
    #[must_use]
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

    /// Returns the array layout.
    #[must_use]
    pub fn layout(&self) -> Layout<D, F> {
        RawSpan::from_buffer(&self.buffer).layout()
    }

    /// Returns the number of elements in the array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns an iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same format.
    /// # Panics
    /// Panics if the rank is not at least 1.
    pub fn outer_iter(&self) -> AxisIter<T, D::Lower, <D::Lower as Dim>::Format<F>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                self.layout().remove_dim(D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same format.
    /// # Panics
    /// Panics if the rank is not at least 1.
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Format<F>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                self.layout().remove_dim(D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns a reformatted array view of the array span.
    /// # Panics
    /// Panics if the array layout is not compatible with the new format.
    #[must_use]
    pub fn reformat<G: Format>(&self) -> ViewArray<T, D, G> {
        self.to_view().into_format()
    }

    /// Returns a mutable reformatted array view of the array span.
    /// # Panics
    /// Panics if the array layout is not compatible with the new format.
    #[must_use]
    pub fn reformat_mut<G: Format>(&mut self) -> ViewArrayMut<T, D, G> {
        self.to_view_mut().into_format()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    #[must_use]
    pub fn reshape<S: Shape>(&self, shape: S) -> ViewArray<T, S::Dim, <S::Dim as Dim>::Format<F>> {
        self.to_view().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    #[must_use]
    pub fn reshape_mut<S: Shape>(
        &mut self,
        shape: S,
    ) -> ViewArrayMut<T, S::Dim, <S::Dim as Dim>::Format<F>> {
        self.to_view_mut().into_shape(shape)
    }

    /// Returns the shape of the array.
    #[must_use]
    pub fn shape(&self) -> D::Shape {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    #[must_use]
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
    }

    /// Divides an array span into two at an index along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    #[must_use]
    pub fn split_at(&self, mid: usize) -> (ViewArray<T, D, F>, ViewArray<T, D, F>) {
        self.to_view().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    #[must_use]
    pub fn split_at_mut(&mut self, mid: usize) -> (ViewArrayMut<T, D, F>, ViewArrayMut<T, D, F>) {
        self.to_view_mut().into_split_at(mid)
    }

    /// Divides an array span into two at an index along the specified dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    #[must_use]
    pub fn split_axis_at<const DIM: usize>(
        &self,
        mid: usize,
    ) -> (
        ViewArray<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
        ViewArray<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view().into_split_axis_at(mid)
    }

    /// Divides a mutable array span into two at an index along the specified dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    #[must_use]
    pub fn split_axis_at_mut<const DIM: usize>(
        &mut self,
        mid: usize,
    ) -> (
        ViewArrayMut<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
        ViewArrayMut<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view_mut().into_split_axis_at(mid)
    }

    /// Returns the distance between elements in the specified dimension.
    #[must_use]
    pub fn stride(&self, dim: usize) -> isize {
        self.layout().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    #[must_use]
    pub fn strides(&self) -> D::Strides {
        self.layout().strides()
    }

    /// Copies the array span into a new array.
    #[cfg(not(feature = "nightly"))]
    #[must_use]
    pub fn to_grid(&self) -> GridArray<T, D>
    where
        T: Clone,
    {
        let mut grid = GridArray::new();

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new array.
    #[cfg(feature = "nightly")]
    #[must_use]
    pub fn to_grid(&self) -> GridArray<T, D>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    #[must_use]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> GridArray<T, D, A>
    where
        T: Clone,
    {
        let mut grid = GridArray::new_in(alloc);

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new vector.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_grid().into_vec()
    }

    /// Copies the array span into a new vector with the specified allocator.
    #[cfg(feature = "nightly")]
    #[must_use]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_grid_in(alloc).into_vec()
    }

    /// Returns an array view of the entire array span.
    #[must_use]
    pub fn to_view(&self) -> ViewArray<T, D, F> {
        unsafe { ViewArray::new_unchecked(self.as_ptr(), self.layout()) }
    }

    /// Returns a mutable array view of the entire array span.
    #[must_use]
    pub fn to_view_mut(&mut self) -> ViewArrayMut<T, D, F> {
        unsafe { ViewArrayMut::new_unchecked(self.as_mut_ptr(), self.layout()) }
    }

    /// Returns an array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[must_use]
    pub fn view<P: Params, I>(&self, index: I) -> ViewArray<T, P::Dim, P::Format>
    where
        I: ViewIndex<D, F, Params = P>,
    {
        self.to_view().into_view(index)
    }

    /// Returns a mutable array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[must_use]
    pub fn view_mut<P: Params, I>(&mut self, index: I) -> ViewArrayMut<T, P::Dim, P::Format>
    where
        I: ViewIndex<D, F, Params = P>,
    {
        self.to_view_mut().into_view(index)
    }
}

impl<T, D: Dim> SpanArray<T, D, Dense> {
    /// Returns a mutable slice of all elements in the array.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T: Clone, D: Dim> ToOwned for SpanArray<T, D, Dense> {
    type Owned = GridArray<T, D>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }
}

fn clone_from_span<T: Clone, D: Dim, F: Format, G: Format>(
    this: &mut SpanArray<T, D, G>,
    src: &SpanArray<T, D, F>,
) {
    if F::IS_UNIFORM && G::IS_UNIFORM {
        assert!(src.shape()[..] == this.shape()[..], "shape mismatch");

        if F::IS_UNIT_STRIDED && G::IS_UNIT_STRIDED {
            this.reformat_mut().as_mut_slice().clone_from_slice(src.reformat().as_slice());
        } else {
            for (x, y) in this.flatten_mut().iter_mut().zip(src.flatten().iter()) {
                x.clone_from(y);
            }
        }
    } else {
        assert!(src.size(D::RANK - 1) == this.size(D::RANK - 1), "shape mismatch");

        for (mut x, y) in this.outer_iter_mut().zip(src.outer_iter()) {
            clone_from_span(&mut x, &y);
        }
    }
}

fn fill_with<T, F: Format>(this: &mut SpanArray<T, impl Dim, F>, f: &mut impl FnMut() -> T) {
    if F::IS_UNIFORM {
        for x in this.flatten_mut().iter_mut() {
            *x = f();
        }
    } else {
        for mut x in this.outer_iter_mut() {
            fill_with(&mut x, f);
        }
    }
}
