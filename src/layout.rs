use std::ops::{Range, RangeBounds};

use crate::dim::{Dim, Shape, U0, U1};
use crate::format::{Dense, Format, General, Linear, Strided, Uniform, UnitStrided};
use crate::order::Order;

use crate::mapping::{DenseMapping, GeneralMapping, LinearMapping, Mapping, StridedMapping};

/// Marker trait for layout types that support linear indexing.
#[marker]
pub trait HasLinearIndexing {}

/// Marker trait for layout types that support slice indexing.
#[marker]
pub trait HasSliceIndexing {}

/// Array layout, including rank, shape, strides and element order.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layout<D: Dim, F: Format, O: Order> {
    map: F::Mapping<D, O>,
}

pub(crate) type DenseLayout<D, O> = Layout<D, Dense, O>;
pub(crate) type GeneralLayout<D, O> = Layout<D, General, O>;
pub(crate) type LinearLayout<D, O> = Layout<D, Linear, O>;
pub(crate) type StridedLayout<D, O> = Layout<D, Strided, O>;

impl<D: Dim, F: Format, O: Order> Layout<D, F, O> {
    /// Returns the dimension with the specified index, counted from the innermost dimension.
    pub fn dim(&self, index: usize) -> usize {
        self.map.dim(index)
    }

    /// Returns the dimensions with the specified indicies, counted from the innermost dimension.
    pub fn dims(&self, indices: impl RangeBounds<usize>) -> Range<usize> {
        self.map.dims(indices)
    }

    /// Returns true if the array layout type supports linear indexing.
    pub fn has_linear_indexing(&self) -> bool {
        self.map.has_linear_indexing()
    }

    /// Returns true if the array layout type supports slice indexing.
    pub fn has_slice_indexing(&self) -> bool {
        self.map.has_slice_indexing()
    }

    /// Returns true if the array has column-major element order.
    pub fn is_column_major(&self) -> bool {
        O::select(true, false)
    }

    /// Returns true if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        self.map.is_contiguous()
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the array has row-major element order.
    pub fn is_row_major(&self) -> bool {
        O::select(false, true)
    }

    /// Returns true if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(&self) -> bool {
        self.map.is_uniformly_strided()
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns the rank of the array.
    pub fn rank(&self) -> usize {
        D::RANK
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> D::Shape {
        self.map.shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.map.size(dim)
    }

    /// Returns the distance between elements in the specified dimension.
    pub fn stride(&self, dim: usize) -> isize {
        self.map.stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> D::Strides {
        self.map.strides()
    }

    pub(crate) fn add_dim(self, size: usize, stride: isize) -> Layout<D::Higher, F, O> {
        self.map.add_dim(size, stride)
    }

    pub(crate) fn flatten(self) -> Layout<U1, F::Uniform, O> {
        self.map.flatten()
    }

    pub(crate) fn offset(&self, index: D::Shape) -> isize {
        self.map.offset(index)
    }

    pub(crate) fn reformat<G: Format>(self) -> Layout<D, G, O> {
        G::Mapping::reformat(self)
    }

    pub(crate) fn remove_dim(self, dim: usize) -> Layout<D::Lower, F, O> {
        self.map.remove_dim(dim)
    }

    pub(crate) fn reshape<S: Shape>(self, shape: S) -> Layout<S::Dim, F, O> {
        self.map.reshape(shape)
    }

    pub(crate) fn resize_dim(self, dim: usize, size: usize) -> Self {
        self.map.resize_dim(dim, size)
    }
}

impl<D: Dim, O: Order> DenseLayout<D, O> {
    /// Creates a new, dense array layout with the specified shape.
    pub fn new(shape: D::Shape) -> Self {
        Self { map: DenseMapping::new(shape) }
    }
}

impl<D: Dim, O: Order> GeneralLayout<D, O> {
    /// Creates a new, general array layout with the specified shape and outer strides.
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        Self { map: GeneralMapping::new(shape, outer_strides) }
    }
}

impl<D: Dim, O: Order> LinearLayout<D, O> {
    /// Creates a new, linear array layout with the specified shape and inner stride.
    pub fn new(shape: D::Shape, inner_stride: <D::MaxOne as Dim>::Strides) -> Self {
        Self { map: LinearMapping::new(shape, inner_stride) }
    }
}

impl<D: Dim, O: Order> StridedLayout<D, O> {
    /// Creates a new, strided array layout with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { map: StridedMapping::new(shape, strides) }
    }
}

impl<D: Dim, F: Uniform, O: Order> HasLinearIndexing for Layout<D, F, O> {}

impl<D: Dim, O: Order> HasSliceIndexing for DenseLayout<D, O> {}

impl<F: Format, O: Order> HasLinearIndexing for Layout<U0, F, O> {}
impl<F: Format, O: Order> HasLinearIndexing for Layout<U1, F, O> {}

impl<F: Format, O: Order> HasSliceIndexing for Layout<U0, F, O> {}

impl<F: UnitStrided, O: Order> HasSliceIndexing for Layout<U1, F, O> {}
