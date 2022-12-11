#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
#[cfg(feature = "nightly")]
use std::marker::PhantomData;
#[cfg(not(feature = "nightly"))]
use std::marker::{PhantomData, PhantomPinned};
use std::ops::{Index, IndexMut};
use std::slice;

use crate::dim::{Dim, Rank, Shape};
use crate::format::{Dense, Format, Uniform};
use crate::grid::{DenseGrid, SubGrid, SubGridMut};
use crate::index::{Axis, Const, Params, SpanIndex, ViewIndex};
use crate::iter::{AxisIter, AxisIterMut};
use crate::layout::Layout;
use crate::mapping::Mapping;
use crate::raw_span::RawSpan;

/// Multidimensional array span with static rank and element order.
pub struct SpanBase<T, D: Dim, F: Format> {
    phantom: PhantomData<(T, D, F)>,
    #[cfg(not(feature = "nightly"))]
    _pinned: PhantomPinned,
    #[cfg(feature = "nightly")]
    _opaque: Opaque,
}

/// Dense multidimensional array span with static rank and element order.
pub type DenseSpan<T, D> = SpanBase<T, D, Dense>;

#[cfg(feature = "nightly")]
extern "C" {
    type Opaque;
}

impl<T, D: Dim, F: Format> SpanBase<T, D, F> {
    /// Returns a mutable pointer to the array buffer.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        RawSpan::from_mut_span(self).as_mut_ptr()
    }

    /// Returns a raw pointer to the array buffer.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        RawSpan::from_span(self).as_ptr()
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
    pub fn clone_from_span(&mut self, src: &SpanBase<T, D, impl Format>)
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
    pub fn flatten(&self) -> SubGrid<T, Rank<1, D::Order>, F::Uniform> {
        self.to_view().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    /// # Panics
    /// Panics if the array layout is not uniformly strided.
    #[must_use]
    pub fn flatten_mut(&mut self) -> SubGridMut<T, Rank<1, D::Order>, F::Uniform> {
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
    #[must_use]
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
    pub fn inner_iter(
        &self,
    ) -> AxisIter<T, D::Lower, <D::Lower as Dim>::Format<F::NonUnitStrided>> {
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
    pub fn inner_iter_mut(
        &mut self,
    ) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Format<F::NonUnitStrided>> {
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
        RawSpan::from_span(self).layout()
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
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Format<F>> {
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
    #[must_use]
    pub fn reformat<G: Format>(&self) -> SubGrid<T, D, G> {
        self.to_view().into_format()
    }

    /// Returns a mutable reformatted array view of the array span.
    /// # Panics
    /// Panics if the array layout is not compatible with the new format.
    #[must_use]
    pub fn reformat_mut<G: Format>(&mut self) -> SubGridMut<T, D, G> {
        self.to_view_mut().into_format()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    #[must_use]
    pub fn reshape<S: Shape>(
        &self,
        shape: S,
    ) -> SubGrid<T, S::Dim<D::Order>, <S::Dim<D::Order> as Dim>::Format<F>> {
        self.to_view().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    /// # Panics
    /// Panics if the array length is changed, or the memory layout is not compatible.
    #[must_use]
    pub fn reshape_mut<S: Shape>(
        &mut self,
        shape: S,
    ) -> SubGridMut<T, S::Dim<D::Order>, <S::Dim<D::Order> as Dim>::Format<F>> {
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
    pub fn split_at(&self, mid: usize) -> (SubGrid<T, D, F>, SubGrid<T, D, F>) {
        self.to_view().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outer dimension.
    /// # Panics
    /// Panics if the split point is larger than the number of elements in that dimension.
    #[must_use]
    pub fn split_at_mut(&mut self, mid: usize) -> (SubGridMut<T, D, F>, SubGridMut<T, D, F>) {
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
        SubGrid<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
        SubGrid<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
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
        SubGridMut<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
        SubGridMut<T, D, <Const<DIM> as Axis<D>>::Split<F>>,
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
    #[must_use]
    pub fn to_grid(&self) -> DenseGrid<T, D>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    #[must_use]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> DenseGrid<T, D, A>
    where
        T: Clone,
    {
        let mut grid = DenseGrid::new_in(alloc);

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
    pub fn to_view(&self) -> SubGrid<T, D, F> {
        unsafe { SubGrid::new_unchecked(self.as_ptr(), self.layout()) }
    }

    /// Returns a mutable array view of the entire array span.
    #[must_use]
    pub fn to_view_mut(&mut self) -> SubGridMut<T, D, F> {
        unsafe { SubGridMut::new_unchecked(self.as_mut_ptr(), self.layout()) }
    }

    /// Returns an array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[must_use]
    pub fn view<P: Params, I>(&self, index: I) -> SubGrid<T, P::Dim, P::Format>
    where
        I: ViewIndex<D, F, Params = P>,
    {
        self.to_view().into_view(index)
    }

    /// Returns a mutable array view for the specified subarray.
    /// # Panics
    /// Panics if the subarray is out of bounds.
    #[must_use]
    pub fn view_mut<P: Params, I>(&mut self, index: I) -> SubGridMut<T, P::Dim, P::Format>
    where
        I: ViewIndex<D, F, Params = P>,
    {
        self.to_view_mut().into_view(index)
    }
}

impl<T, D: Dim> DenseSpan<T, D> {
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

impl<T: Debug, D: Dim, F: Format> Debug for SpanBase<T, D, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if D::RANK == 0 {
            self[D::Shape::default()].fmt(f)
        } else {
            let mut list = f.debug_list();

            if !self.is_empty() {
                if D::RANK == 1 {
                    list.entries(self.flatten().iter());
                } else {
                    list.entries(self.outer_iter());
                }
            }

            list.finish()
        }
    }
}

impl<T: Hash, D: Dim, F: Format> Hash for SpanBase<T, D, F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let shape = if self.is_empty() { Default::default() } else { self.shape() };

        for i in 0..D::RANK {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(shape[D::dim(i)]);
            #[cfg(feature = "nightly")]
            state.write_length_prefix(shape[D::dim(i)]);
        }

        hash(self, state);
    }
}

impl<T, D: Dim, F: Format, I: SpanIndex<T, D, F>> Index<I> for SpanBase<T, D, F> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, D: Dim, F: Format, I: SpanIndex<T, D, F>> IndexMut<I> for SpanBase<T, D, F> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, D: Dim, F: Uniform> IntoIterator for &'a SpanBase<T, D, F> {
    type Item = &'a T;
    type IntoIter = F::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: Dim, F: Uniform> IntoIterator for &'a mut SpanBase<T, D, F> {
    type Item = &'a mut T;
    type IntoIter = F::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

unsafe impl<T: Send, D: Dim, F: Format> Send for SpanBase<T, D, F> {}
unsafe impl<T: Sync, D: Dim, F: Format> Sync for SpanBase<T, D, F> {}

impl<T: Clone, D: Dim> ToOwned for DenseSpan<T, D> {
    type Owned = DenseGrid<T, D>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }
}

fn clone_from_span<T: Clone, D: Dim, F: Format, G: Format>(
    this: &mut SpanBase<T, D, G>,
    src: &SpanBase<T, D, F>,
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
        let dim = D::dim(D::RANK - 1);

        assert!(src.size(dim) == this.size(dim), "shape mismatch");

        for (mut x, y) in this.outer_iter_mut().zip(src.outer_iter()) {
            clone_from_span(&mut x, &y);
        }
    }
}

fn fill_with<T, F: Format>(this: &mut SpanBase<T, impl Dim, F>, f: &mut impl FnMut() -> T) {
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

fn hash<T: Hash, F: Format>(this: &SpanBase<T, impl Dim, F>, state: &mut impl Hasher) {
    if F::IS_UNIFORM {
        for x in this.flatten().iter() {
            x.hash(state);
        }
    } else {
        for x in this.outer_iter() {
            hash(&x, state);
        }
    }
}
