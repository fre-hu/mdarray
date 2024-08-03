use std::fmt::Debug;

use crate::layout::{Dense, Flat, General, Layout, Strided};
use crate::shape::Shape;

/// Array layout mapping trait, including shape and strides.
pub trait Mapping: Copy + Debug + Default + Send + Sync {
    /// Array shape type.
    type Shape: Shape;

    /// Array layout type.
    type Layout: Layout<Mapping<Self::Shape> = Self>;

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    fn is_contiguous(self) -> bool;

    /// Returns `true` if the array strides are consistent with uniformly strided memory layout.
    fn is_uniformly_strided(self) -> bool;

    /// Returns the array shape.
    fn shape(self) -> Self::Shape;

    /// Returns the distance between elements in each dimension.
    fn strides(self) -> <Self::Shape as Shape>::Strides;

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(self, index: usize) -> usize {
        self.shape().dim(index)
    }

    /// Returns the number of elements in each dimension.
    fn dims(self) -> <Self::Shape as Shape>::Dims {
        self.shape().dims()
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(self) -> bool {
        self.shape().is_empty()
    }

    /// Returns the number of elements in the array.
    fn len(self) -> usize {
        self.shape().len()
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(self) -> usize {
        Self::Shape::RANK
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn stride(self, index: usize) -> isize {
        assert!(index < Self::Shape::RANK, "invalid dimension");

        self.strides()[index]
    }

    #[doc(hidden)]
    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self;

    #[doc(hidden)]
    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self;

    #[doc(hidden)]
    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self;

    #[doc(hidden)]
    fn reshape<M: Mapping>(mapping: M, new_shape: Self::Shape) -> Self;

    #[doc(hidden)]
    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self;

    #[doc(hidden)]
    fn offset(self, index: <Self::Shape as Shape>::Dims) -> isize {
        let dims = self.dims();
        let strides = self.strides();

        let mut offset = 0;

        for i in 0..Self::Shape::RANK {
            debug_assert!(index[i] < dims[i], "index out of bounds");

            offset += strides[i] * index[i] as isize;
        }

        offset
    }
}

/// Dense layout mapping type.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseMapping<S: Shape> {
    shape: S,
}

/// Flat layout mapping type.
#[derive(Clone, Copy, Debug)]
pub struct FlatMapping<S: Shape> {
    shape: S,
    inner_stride: isize,
}

/// General layout mapping type.
#[derive(Clone, Copy, Debug)]
pub struct GeneralMapping<S: Shape> {
    shape: S,
    outer_strides: <S::Tail as Shape>::Strides,
}

/// Strided layout mapping type.
#[derive(Clone, Copy, Debug)]
pub struct StridedMapping<S: Shape> {
    shape: S,
    strides: S::Strides,
}

impl<S: Shape> DenseMapping<S> {
    /// Creates a new, dense layout mapping with the specified shape.
    pub fn new(shape: S) -> Self {
        Self { shape }
    }
}

impl<S: Shape> Mapping for DenseMapping<S> {
    type Shape = S;
    type Layout = Dense;

    fn is_contiguous(self) -> bool {
        true
    }

    fn is_uniformly_strided(self) -> bool {
        true
    }

    fn shape(self) -> S {
        self.shape
    }

    fn strides(self) -> S::Strides {
        let mut strides = S::Strides::default();
        let mut stride = 1;

        for i in 0..S::RANK {
            strides[i] = stride as isize;
            stride *= self.dim(i);
        }

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(M::Layout::IS_UNIFORM && M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(stride == mapping.len() as isize, "invalid stride");

        Self::new(mapping.shape().add_dim(size))
    }

    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");

        Self::new(mapping.shape())
    }

    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self {
        assert!(M::Shape::RANK < 2 || M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(M::Shape::RANK < 3 || M::Layout::IS_UNIFORM, "invalid layout");
        assert!(index + 1 == M::Shape::RANK, "invalid dimension");

        Self::new(mapping.shape().remove_dim(index))
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: Self::Shape) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");
        assert!(new_shape.checked_len() == Some(mapping.len()), "length must not change");

        Self::new(new_shape)
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        assert!(M::Layout::IS_UNIFORM && M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(index + 1 == M::Shape::RANK, "invalid dimension");

        Self::new(mapping.shape().resize_dim(index, new_size))
    }
}

impl<S: Shape> FlatMapping<S> {
    /// Creates a new, flat layout mapping with the specified shape and inner stride.
    pub fn new(shape: S, inner_stride: isize) -> Self {
        assert!(S::RANK > 0, "invalid rank");

        Self { shape, inner_stride }
    }
}

impl<S: Shape> Default for FlatMapping<S> {
    fn default() -> Self {
        Self::new(S::default(), 1)
    }
}

impl<S: Shape> Mapping for FlatMapping<S> {
    type Shape = S;
    type Layout = Flat;

    fn is_contiguous(self) -> bool {
        self.inner_stride == 1
    }

    fn is_uniformly_strided(self) -> bool {
        true
    }

    fn shape(self) -> S {
        self.shape
    }

    fn strides(self) -> S::Strides {
        let mut strides = S::Strides::default();
        let mut stride = self.inner_stride;

        for i in 0..S::RANK {
            strides[i] = stride;
            stride *= self.dim(i) as isize;
        }

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(M::Layout::IS_UNIFORM, "invalid layout");

        let inner_stride = if M::Shape::RANK > 0 { mapping.stride(0) } else { stride };

        assert!(stride == inner_stride * mapping.len() as isize, "invalid stride");

        Self::new(mapping.shape().add_dim(size), inner_stride)
    }

    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self {
        assert!(M::Shape::RANK > 0, "invalid rank");
        assert!(mapping.is_uniformly_strided(), "mapping not uniformly strided");

        Self::new(mapping.shape(), mapping.stride(0))
    }

    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self {
        assert!(M::Shape::RANK > 1, "invalid rank");
        assert!(M::Shape::RANK < 3 || M::Layout::IS_UNIFORM, "invalid layout");
        assert!(index == 0 || index + 1 == M::Shape::RANK, "invalid dimension");

        let inner_stride = if index > 0 { mapping.stride(0) } else { mapping.stride(1) };

        Self::new(mapping.shape().remove_dim(index), inner_stride)
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: Self::Shape) -> Self {
        assert!(mapping.is_uniformly_strided(), "mapping not uniformly strided");
        assert!(new_shape.checked_len() == Some(mapping.len()), "length must not change");

        Self::new(new_shape, if M::Shape::RANK > 0 { mapping.stride(0) } else { 1 })
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        assert!(M::Layout::IS_UNIFORM, "invalid layout");
        assert!(index + 1 == M::Shape::RANK, "invalid dimension");

        Self::new(mapping.shape().resize_dim(index, new_size), mapping.stride(0))
    }
}

impl<S: Shape> GeneralMapping<S> {
    /// Creates a new, general layout mapping with the specified shape and outer strides.
    pub fn new(shape: S, outer_strides: <S::Tail as Shape>::Strides) -> Self {
        assert!(S::RANK > 1, "invalid rank");

        Self { shape, outer_strides }
    }
}

impl<S: Shape> Default for GeneralMapping<S> {
    fn default() -> Self {
        Self::new(S::default(), Default::default())
    }
}

impl<S: Shape> Mapping for GeneralMapping<S> {
    type Shape = S;
    type Layout = General;

    fn is_contiguous(self) -> bool {
        let mut stride = self.dim(0);

        for i in 1..S::RANK {
            if self.outer_strides[i - 1] != stride as isize {
                return false;
            }

            stride *= self.dim(i);
        }

        true
    }

    fn is_uniformly_strided(self) -> bool {
        self.is_contiguous()
    }

    fn shape(self) -> S {
        self.shape
    }

    fn stride(self, dim: usize) -> isize {
        assert!(dim < S::RANK, "invalid dimension");

        if dim > 0 {
            self.outer_strides[dim - 1]
        } else {
            1
        }
    }

    fn strides(self) -> S::Strides {
        let mut strides = S::Strides::default();

        strides[0] = 1;
        strides[1..].copy_from_slice(&self.outer_strides[..]);

        strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");

        Self::remap(StridedMapping::add_dim(mapping, size, stride))
    }

    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self {
        assert!(M::Shape::RANK > 1, "invalid rank");
        assert!(mapping.stride(0) == 1, "inner stride not unitary");

        let outer_strides = TryFrom::try_from(&mapping.strides()[1..]).expect("invalid rank");

        Self::new(mapping.shape(), outer_strides)
    }

    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");
        assert!(index > 0, "invalid dimension");

        Self::remap(StridedMapping::remove_dim(mapping, index))
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: Self::Shape) -> Self {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");

        Self::remap(StridedMapping::reshape(mapping, new_shape))
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        assert!(M::Layout::IS_UNIT_STRIDED, "invalid layout");

        Self::remap(StridedMapping::resize_dim(mapping, index, new_size))
    }
}

impl<S: Shape> StridedMapping<S> {
    /// Creates a new, strided layout mapping with the specified shape and strides.
    pub fn new(shape: S, strides: S::Strides) -> Self {
        assert!(S::RANK > 1, "invalid rank");

        Self { shape, strides }
    }
}

impl<S: Shape> Default for StridedMapping<S> {
    fn default() -> Self {
        assert!(S::RANK > 1, "invalid rank");

        let mut strides = S::Strides::default();

        strides[0] = 1;

        Self { shape: S::default(), strides }
    }
}

impl<S: Shape> Mapping for StridedMapping<S> {
    type Shape = S;
    type Layout = Strided;

    fn is_contiguous(self) -> bool {
        self.strides[0] == 1 && self.is_uniformly_strided()
    }

    fn is_uniformly_strided(self) -> bool {
        let mut stride = self.strides[0];

        for i in 1..S::RANK {
            stride *= self.dim(i - 1) as isize;

            if self.strides[i] != stride {
                return false;
            }
        }

        true
    }

    fn shape(self) -> S {
        self.shape
    }

    fn strides(self) -> S::Strides {
        self.strides
    }

    fn add_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(S::RANK == M::Shape::RANK + 1, "invalid rank");

        let mut strides = S::Strides::default();

        strides[..M::Shape::RANK].copy_from_slice(&mapping.strides()[..]);
        strides[M::Shape::RANK] = stride;

        Self::new(mapping.shape().add_dim(size), strides)
    }

    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self {
        Self::new(mapping.shape(), mapping.strides())
    }

    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self {
        assert!(S::RANK + 1 == M::Shape::RANK, "invalid rank");
        assert!(index < M::Shape::RANK, "invalid dimension");

        let mut strides = S::Strides::default();

        strides[..index].copy_from_slice(&mapping.strides()[..index]);
        strides[index..].copy_from_slice(&mapping.strides()[index + 1..]);

        Self::new(mapping.shape().remove_dim(index), strides)
    }

    fn reshape<M: Mapping>(mapping: M, new_shape: Self::Shape) -> Self {
        assert!(new_shape.checked_len() == Some(mapping.len()), "length must not change");

        let old_dims = mapping.dims();
        let new_dims = new_shape.dims();

        let old_strides = mapping.strides();
        let mut new_strides = S::Strides::default();

        let mut old_len = 1usize;
        let mut new_len = 1usize;

        let mut old_stride = 1;
        let mut new_stride = 1;

        let mut valid_layout = true;

        let mut k = 0;

        for i in 0..M::Shape::RANK {
            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = old_strides[i];
                new_stride = old_stride;
            } else {
                valid_layout &= old_stride == old_strides[i];
            }

            old_len *= old_dims[i];
            old_stride *= old_dims[i] as isize;

            // Add dimensions within the current region.
            while k < S::RANK {
                let len = new_len * new_dims[k];

                if len > old_len {
                    break;
                }

                new_strides[k] = new_stride;

                new_len = len;
                new_stride *= new_dims[k] as isize;

                k += 1;
            }
        }

        // Add remaining dimensions.
        while k < S::RANK {
            new_strides[k] = new_stride;

            new_len *= new_dims[k];
            new_stride *= new_dims[k] as isize;

            k += 1;
        }

        assert!(new_len == 0 || valid_layout, "memory layout not compatible");

        Self::new(new_shape, new_strides)
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        let strides = TryFrom::try_from(&mapping.strides()[..]).expect("invalid rank");

        Self::new(mapping.shape().resize_dim(index, new_size), strides)
    }
}
