use crate::dim::{Dim, Shape, U1};
use crate::format::{Dense, Flat, Format, General, Strided};
use crate::index::Params;
use crate::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};
use crate::order::Order;

/// Array layout, including rank, shape, strides and element order.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layout<D: Dim, F: Format, O: Order> {
    map: F::Mapping<D, O>,
}

pub type DenseLayout<D, O> = Layout<D, Dense, O>;
pub type FlatLayout<D, O> = Layout<D, Flat, O>;
pub type GeneralLayout<D, O> = Layout<D, General, O>;
pub type StridedLayout<D, O> = Layout<D, Strided, O>;

pub type ValidLayout<D, F, O> = Layout<D, <D as Dim>::Format<F>, O>;
pub type ViewLayout<P, O> = Layout<<P as Params>::Dim, <P as Params>::Format, O>;

impl<D: Dim, F: Format, O: Order> Layout<D, F, O> {
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

    pub(crate) fn add_dim<G: Format>(self, size: usize, stride: isize) -> Layout<D::Higher, G, O> {
        G::Mapping::add_dim(self, size, stride)
    }

    pub(crate) fn flatten(self) -> Layout<U1, F::Uniform, O> {
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

    pub(crate) fn reformat<G: Format>(self) -> Layout<D, G, O> {
        G::Mapping::reformat(self)
    }

    pub(crate) fn remove_dim<G: Format>(self, dim: usize) -> Layout<D::Lower, G, O> {
        G::Mapping::remove_dim(self, dim)
    }

    pub(crate) fn reshape<S: Shape, G: Format>(self, new_shape: S) -> Layout<S::Dim, G, O> {
        G::Mapping::reshape(self, new_shape)
    }

    pub(crate) fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        self.map.resize_dim(dim, new_size)
    }
}

impl<D: Dim, O: Order> DenseLayout<D, O> {
    /// Creates a new, dense array layout with the specified shape.
    pub fn new(shape: D::Shape) -> Self {
        Self { map: DenseMapping::new(shape) }
    }
}

impl<D: Dim, O: Order> FlatLayout<D, O> {
    /// Creates a new, flat array layout with the specified shape and inner stride.
    pub fn new(shape: D::Shape, inner_stride: isize) -> Self {
        Self { map: FlatMapping::new(shape, inner_stride) }
    }
}

impl<D: Dim, O: Order> GeneralLayout<D, O> {
    /// Creates a new, general array layout with the specified shape and outer strides.
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        Self { map: GeneralMapping::new(shape, outer_strides) }
    }
}

impl<D: Dim, O: Order> StridedLayout<D, O> {
    /// Creates a new, strided array layout with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { map: StridedMapping::new(shape, strides) }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
pub fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}
