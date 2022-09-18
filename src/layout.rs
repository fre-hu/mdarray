use std::fmt::{Debug, Formatter, Result};

use crate::dim::{Dim, Rank, Shape};
use crate::format::{Dense, Flat, Format, General, Strided};
use crate::index::Params;
use crate::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};

/// Array layout, including rank, element order and storage format.
pub struct Layout<D: Dim, F: Format> {
    map: F::Mapping<D>,
}

pub type DenseLayout<D> = Layout<D, Dense>;
pub type FlatLayout<D> = Layout<D, Flat>;
pub type GeneralLayout<D> = Layout<D, General>;
pub type StridedLayout<D> = Layout<D, Strided>;

pub type ValidLayout<D, F> = Layout<D, <D as Dim>::Format<F>>;
pub type ViewLayout<P> = Layout<<P as Params>::Dim, <P as Params>::Format>;

impl<D: Dim, F: Format> Layout<D, F> {
    /// Returns true if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(self) -> bool {
        self.map.is_contiguous()
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns true if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(self) -> bool {
        self.map.is_uniformly_strided()
    }

    /// Returns the number of elements in the array.
    pub fn len(self) -> usize {
        self.map.len()
    }

    /// Returns the shape of the array.
    pub fn shape(self) -> D::Shape {
        self.map.shape()
    }

    /// Returns the number of elements in the specified dimension.
    /// # Panics
    /// Panics if the dimension is out of bounds.
    pub fn size(self, dim: usize) -> usize {
        assert!(dim < D::RANK, "invalid dimension");

        self.map.shape()[dim]
    }

    /// Returns the distance between elements in the specified dimension.
    /// # Panics
    /// Panics if the dimension is out of bounds.
    pub fn stride(self, dim: usize) -> isize {
        assert!(dim < D::RANK, "invalid dimension");

        self.map.strides()[dim]
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(self) -> D::Strides {
        self.map.strides()
    }

    pub(crate) fn add_dim<G: Format>(self, size: usize, stride: isize) -> Layout<D::Higher, G> {
        G::Mapping::add_dim(self, size, stride)
    }

    pub(crate) fn flatten(self) -> Layout<Rank<1, D::Order>, F::Uniform> {
        self.map.flatten()
    }

    pub(crate) fn offset(self, index: D::Shape) -> isize {
        let mut offset = 0;
        let strides = self.map.strides();

        for i in 0..D::RANK {
            debug_assert!(index[i] < self.map.shape()[i], "index out of bounds");

            offset += strides[i] * index[i] as isize;
        }

        offset
    }

    pub(crate) fn reformat<G: Format>(self) -> Layout<D, G> {
        G::Mapping::reformat(self)
    }

    pub(crate) fn remove_dim<G: Format>(self, dim: usize) -> Layout<D::Lower, G> {
        G::Mapping::remove_dim(self, dim)
    }

    pub(crate) fn reshape<S: Shape, G: Format>(self, new_shape: S) -> Layout<S::Dim<D::Order>, G> {
        G::Mapping::reshape(self, new_shape)
    }

    pub(crate) fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        self.map.resize_dim(dim, new_size)
    }
}

impl<D: Dim> DenseLayout<D> {
    /// Creates a new, dense array layout with the specified shape.
    pub fn new(shape: D::Shape) -> Self {
        Self { map: DenseMapping::new(shape) }
    }
}

impl<D: Dim> FlatLayout<D> {
    /// Creates a new, flat array layout with the specified shape and inner stride.
    pub fn new(shape: D::Shape, inner_stride: isize) -> Self {
        Self { map: FlatMapping::new(shape, inner_stride) }
    }
}

impl<D: Dim> GeneralLayout<D> {
    /// Creates a new, general array layout with the specified shape and outer strides.
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        Self { map: GeneralMapping::new(shape, outer_strides) }
    }
}

impl<D: Dim> StridedLayout<D> {
    /// Creates a new, strided array layout with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { map: StridedMapping::new(shape, strides) }
    }
}

impl<D: Dim, F: Format> Clone for Layout<D, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dim, F: Format> Copy for Layout<D, F> {}

impl<D: Dim, F: Format> Debug for Layout<D, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.map.fmt(f)
    }
}

impl<D: Dim, F: Format> Default for Layout<D, F> {
    fn default() -> Self {
        Self { map: Default::default() }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
pub fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}
