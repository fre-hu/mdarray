use std::fmt::{Debug, Formatter, Result};

use crate::dim::Dim;
use crate::layout::{Dense, Flat, General, Layout, Strided};

/// Array layout mapping trait, including shape and strides.
pub trait Mapping: Copy + Debug + Default {
    /// Array dimension type.
    type Dim: Dim;

    /// Array layout type.
    type Layout: Layout<Mapping<Self::Dim> = Self>;

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    fn is_contiguous(self) -> bool;

    /// Returns `true` if the array strides are consistent with uniformly strided memory layout.
    fn is_uniformly_strided(self) -> bool;

    /// Returns the shape of the array.
    fn shape(self) -> <Self::Dim as Dim>::Shape;

    /// Returns the distance between elements in each dimension.
    fn strides(self) -> <Self::Dim as Dim>::Strides;

    /// Returns `true` if the array contains no elements.
    fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(self) -> usize {
        self.shape()[..].iter().product()
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(self) -> usize {
        Self::Dim::RANK
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn size(self, dim: usize) -> usize {
        assert!(dim < Self::Dim::RANK, "invalid dimension");

        self.shape()[dim]
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn stride(self, dim: usize) -> isize {
        assert!(dim < Self::Dim::RANK, "invalid dimension");

        self.strides()[dim]
    }

    #[doc(hidden)]
    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self
    where
        M::Dim: Dim<Higher = Self::Dim>;

    #[doc(hidden)]
    fn remap<M: Mapping<Dim = Self::Dim>>(mapping: M) -> Self;

    #[doc(hidden)]
    fn remove_dim<M: Mapping>(mapping: M, dim: usize) -> Self
    where
        M::Dim: Dim<Lower = Self::Dim>;

    #[doc(hidden)]
    fn reshape<M: Mapping>(mapping: M, new_shape: <Self::Dim as Dim>::Shape) -> Self;

    #[doc(hidden)]
    fn resize_dim(self, dim: usize, new_size: usize) -> Self;

    #[doc(hidden)]
    fn offset(self, index: <Self::Dim as Dim>::Shape) -> isize {
        let shape = self.shape();
        let strides = self.strides();

        let mut offset = 0;

        for i in 0..Self::Dim::RANK {
            debug_assert!(index[i] < shape[i], "index out of bounds");

            offset += strides[i] * index[i] as isize;
        }

        offset
    }
}

/// Dense layout mapping type.
pub struct DenseMapping<D: Dim> {
    shape: D::Shape,
}

/// Flat layout mapping type.
pub struct FlatMapping<D: Dim> {
    shape: D::Shape,
    inner_stride: isize,
}

/// General layout mapping type.
pub struct GeneralMapping<D: Dim> {
    shape: D::Shape,
    outer_strides: <D::Lower as Dim>::Strides,
}

/// Strided layout mapping type.
pub struct StridedMapping<D: Dim> {
    shape: D::Shape,
    strides: D::Strides,
}

impl<D: Dim> DenseMapping<D> {
    /// Creates a new, dense layout mapping with the specified shape.
    pub fn new(shape: D::Shape) -> Self {
        Self { shape }
    }
}

impl<D: Dim> Clone for DenseMapping<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dim> Copy for DenseMapping<D> {}

impl<D: Dim> Debug for DenseMapping<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("DenseMapping").field("shape", &self.shape).finish()
    }
}

impl<D: Dim> Default for DenseMapping<D> {
    fn default() -> Self {
        Self { shape: Default::default() }
    }
}

impl<D: Dim> Mapping for DenseMapping<D> {
    type Dim = D;
    type Layout = Dense;

    fn is_contiguous(self) -> bool {
        true
    }

    fn is_uniformly_strided(self) -> bool {
        true
    }

    fn shape(self) -> D::Shape {
        self.shape
    }

    fn strides(self) -> D::Strides {
        let mut strides = D::Strides::default();
        let mut stride = 1;

        for i in 0..D::RANK {
            strides[i] = stride as isize;
            stride *= self.shape[i];
        }

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self
    where
        M::Dim: Dim<Higher = D>,
    {
        assert!(M::Layout::IS_UNIFORM && M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(stride == mapping.len() as isize, "invalid stride");

        Self::new(M::Dim::add_dim(mapping.shape(), size))
    }

    fn remap<M: Mapping<Dim = D>>(mapping: M) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");

        Self::new(mapping.shape())
    }

    fn remove_dim<M: Mapping>(mapping: M, dim: usize) -> Self
    where
        M::Dim: Dim<Lower = D>,
    {
        assert!(D::RANK < 1 || M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(D::RANK < 2 || M::Layout::IS_UNIFORM, "invalid layout");
        assert!(dim == D::RANK, "invalid dimension");

        Self::new(M::Dim::remove_dim(mapping.shape(), dim))
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: D::Shape) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");
        assert!(D::checked_len(new_shape) == mapping.len(), "length must not change");

        Self::new(new_shape)
    }

    fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        assert!(dim + 1 == D::RANK, "invalid dimension");

        Self::new(D::resize_dim(self.shape, dim, new_size))
    }
}

impl<D: Dim> FlatMapping<D> {
    /// Creates a new, flat layout mapping with the specified shape and inner stride.
    pub fn new(shape: D::Shape, inner_stride: isize) -> Self {
        assert!(D::RANK > 0, "invalid rank");

        Self { shape, inner_stride }
    }
}

impl<D: Dim> Clone for FlatMapping<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dim> Copy for FlatMapping<D> {}

impl<D: Dim> Debug for FlatMapping<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("FlatMapping")
            .field("shape", &self.shape)
            .field("inner_stride", &self.inner_stride)
            .finish()
    }
}

impl<D: Dim> Default for FlatMapping<D> {
    fn default() -> Self {
        assert!(D::RANK > 0, "invalid rank");

        Self { shape: Default::default(), inner_stride: 1 }
    }
}

impl<D: Dim> Mapping for FlatMapping<D> {
    type Dim = D;
    type Layout = Flat;

    fn is_contiguous(self) -> bool {
        self.inner_stride == 1
    }

    fn is_uniformly_strided(self) -> bool {
        true
    }

    fn shape(self) -> D::Shape {
        self.shape
    }

    fn strides(self) -> D::Strides {
        let mut strides = D::Strides::default();
        let mut stride = self.inner_stride;

        for i in 0..D::RANK {
            strides[i] = stride;
            stride *= self.shape[i] as isize;
        }

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self
    where
        M::Dim: Dim<Higher = D>,
    {
        assert!(M::Layout::IS_UNIFORM, "invalid layout");

        let inner_stride = if M::Dim::RANK > 0 { mapping.stride(0) } else { stride };

        assert!(stride == inner_stride * mapping.len() as isize, "invalid stride");

        Self::new(M::Dim::add_dim(mapping.shape(), size), inner_stride)
    }

    fn remap<M: Mapping<Dim = D>>(mapping: M) -> Self {
        assert!(D::RANK > 0, "invalid rank");
        assert!(mapping.is_uniformly_strided(), "mapping not uniformly strided");

        Self::new(mapping.shape(), mapping.stride(0))
    }

    fn remove_dim<M: Mapping>(mapping: M, dim: usize) -> Self
    where
        M::Dim: Dim<Lower = D>,
    {
        assert!(D::RANK > 0, "invalid rank");
        assert!(D::RANK < 2 || M::Layout::IS_UNIFORM, "invalid layout");
        assert!(dim == 0 || dim == D::RANK, "invalid dimension");

        let inner_stride = if dim > 0 { mapping.stride(0) } else { mapping.stride(1) };

        Self::new(M::Dim::remove_dim(mapping.shape(), dim), inner_stride)
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: D::Shape) -> Self {
        assert!(mapping.is_uniformly_strided(), "mapping not uniformly strided");
        assert!(D::checked_len(new_shape) == mapping.len(), "length must not change");

        Self::new(new_shape, if M::Dim::RANK > 0 { mapping.stride(0) } else { 1 })
    }

    fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        assert!(dim + 1 == D::RANK, "invalid dimension");

        Self::new(D::resize_dim(self.shape, dim, new_size), self.inner_stride)
    }
}

impl<D: Dim> GeneralMapping<D> {
    /// Creates a new, general layout mapping with the specified shape and outer strides.
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        assert!(D::RANK > 1, "invalid rank");

        Self { shape, outer_strides }
    }
}

impl<D: Dim> Clone for GeneralMapping<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dim> Copy for GeneralMapping<D> {}

impl<D: Dim> Debug for GeneralMapping<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("GeneralMapping")
            .field("shape", &self.shape)
            .field("outer_strides", &self.outer_strides)
            .finish()
    }
}

impl<D: Dim> Default for GeneralMapping<D> {
    fn default() -> Self {
        assert!(D::RANK > 1, "invalid rank");

        Self { shape: Default::default(), outer_strides: Default::default() }
    }
}

impl<D: Dim> Mapping for GeneralMapping<D> {
    type Dim = D;
    type Layout = General;

    fn is_contiguous(self) -> bool {
        let mut stride = self.shape[0];

        for i in 1..D::RANK {
            if self.outer_strides[i - 1] != stride as isize {
                return false;
            }

            stride *= self.shape[i];
        }

        true
    }

    fn is_uniformly_strided(self) -> bool {
        self.is_contiguous()
    }

    fn shape(self) -> D::Shape {
        self.shape
    }

    fn strides(self) -> D::Strides {
        let mut strides = D::Strides::default();

        strides[0] = 1;
        strides[1..].copy_from_slice(&self.outer_strides[..]);

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self
    where
        M::Dim: Dim<Higher = D>,
    {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");

        Self::remap(StridedMapping::add_dim(mapping, size, stride))
    }

    fn remap<M: Mapping<Dim = D>>(mapping: M) -> Self {
        assert!(D::RANK > 1, "invalid rank");
        assert!(mapping.stride(0) == 1, "inner stride not unitary");

        let mut outer_strides = <D::Lower as Dim>::Strides::default();

        outer_strides[..].copy_from_slice(&mapping.strides()[1..]);

        Self::new(mapping.shape(), outer_strides)
    }

    fn remove_dim<M: Mapping>(mapping: M, dim: usize) -> Self
    where
        M::Dim: Dim<Lower = D>,
    {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(dim > 0, "invalid dimension");

        Self::remap(StridedMapping::remove_dim(mapping, dim))
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: D::Shape) -> Self {
        Self::remap(StridedMapping::reshape(mapping, new_shape))
    }

    fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        Self::new(D::resize_dim(self.shape, dim, new_size), self.outer_strides)
    }
}

impl<D: Dim> StridedMapping<D> {
    /// Creates a new, strided layout mapping with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        assert!(D::RANK > 1, "invalid rank");

        Self { shape, strides }
    }
}

impl<D: Dim> Clone for StridedMapping<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dim> Copy for StridedMapping<D> {}

impl<D: Dim> Debug for StridedMapping<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("StridedMapping")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

impl<D: Dim> Default for StridedMapping<D> {
    fn default() -> Self {
        assert!(D::RANK > 1, "invalid rank");

        let mut strides = D::Strides::default();

        strides[0] = 1;

        Self { shape: Default::default(), strides }
    }
}

impl<D: Dim> Mapping for StridedMapping<D> {
    type Dim = D;
    type Layout = Strided;

    fn is_contiguous(self) -> bool {
        self.strides[0] == 1 && self.is_uniformly_strided()
    }

    fn is_uniformly_strided(self) -> bool {
        let mut stride = self.strides[0];

        for i in 1..D::RANK {
            stride *= self.shape[i - 1] as isize;

            if self.strides[i] != stride {
                return false;
            }
        }

        true
    }

    fn shape(self) -> D::Shape {
        self.shape
    }

    fn strides(self) -> D::Strides {
        self.strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self
    where
        M::Dim: Dim<Higher = D>,
    {
        assert!(D::RANK > M::Dim::RANK, "invalid rank");

        let mut strides = D::Strides::default();

        strides[..D::RANK - 1].copy_from_slice(&mapping.strides()[..]);
        strides[D::RANK - 1] = stride;

        Self::new(M::Dim::add_dim(mapping.shape(), size), strides)
    }

    fn remap<M: Mapping<Dim = D>>(mapping: M) -> Self {
        Self::new(mapping.shape(), mapping.strides())
    }

    fn remove_dim<M: Mapping>(mapping: M, dim: usize) -> Self
    where
        M::Dim: Dim<Lower = D>,
    {
        assert!(D::RANK > 1, "invalid rank");
        assert!(dim <= D::RANK, "invalid dimension");

        let mut strides = D::Strides::default();

        strides[..dim].copy_from_slice(&mapping.strides()[..dim]);
        strides[dim..].copy_from_slice(&mapping.strides()[dim + 1..]);

        Self::new(M::Dim::remove_dim(mapping.shape(), dim), strides)
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: D::Shape) -> Self {
        let old_shape = mapping.shape();
        let old_strides = mapping.strides();

        let mut new_strides = D::Strides::default();

        let mut old_len = 1usize;
        let mut new_len = 1usize;

        let mut old_stride = 1;
        let mut new_stride = 1;

        let mut k = 0;

        for i in 0..M::Dim::RANK {
            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = old_strides[i];
                new_stride = old_stride;
            } else {
                assert!(old_stride == old_strides[i], "memory layout not compatible");
            }

            old_len *= old_shape[i];
            old_stride *= old_shape[i] as isize;

            // Add dimensions within the current region.
            while k < D::RANK {
                let len = new_len.checked_mul(new_shape[k]).expect("length too large");

                if len > old_len {
                    break;
                }

                new_strides[k] = new_stride;

                new_len = len;
                new_stride *= new_shape[k] as isize;

                k += 1;
            }
        }

        // Add remaining dimensions.
        while k < D::RANK {
            new_strides[k] = new_stride;

            new_len = new_len.checked_mul(new_shape[k]).expect("length too large");
            new_stride *= new_shape[k] as isize;

            k += 1;
        }

        assert!(new_len == old_len, "length must not change");

        Self::new(new_shape, new_strides)
    }

    fn resize_dim(self, dim: usize, new_size: usize) -> Self {
        Self::new(D::resize_dim(self.shape, dim, new_size), self.strides)
    }
}
