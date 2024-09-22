use std::fmt::Debug;

use crate::layout::{Dense, Layout, Strided};
use crate::shape::Shape;

/// Array layout mapping trait, including shape and strides.
pub trait Mapping: Copy + Debug + Default + Send + Sync {
    /// Array shape type.
    type Shape: Shape;

    /// Array layout type.
    type Layout: Layout<Mapping<Self::Shape> = Self>;

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    fn is_contiguous(self) -> bool;

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
    fn linear_offset(self, index: usize) -> isize;

    #[doc(hidden)]
    fn prepend_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self;

    #[doc(hidden)]
    fn remap<M: Mapping<Shape = Self::Shape>>(mapping: M) -> Self;

    #[doc(hidden)]
    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self;

    #[doc(hidden)]
    fn reorder<M: Mapping<Shape = <Self::Shape as Shape>::Reverse>>(mapping: M) -> Self;

    #[doc(hidden)]
    fn reshape<M: Mapping<Layout = Self::Layout>>(mapping: M, new_shape: Self::Shape) -> Self;

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

    fn shape(self) -> S {
        self.shape
    }

    fn strides(self) -> S::Strides {
        let mut strides = S::Strides::default();
        let mut stride = 1;

        for i in (0..S::RANK).rev() {
            strides[i] = stride as isize;
            stride *= self.dim(i);
        }

        strides
    }

    fn linear_offset(self, index: usize) -> isize {
        debug_assert!(index < self.len(), "index out of bounds");

        index as isize
    }

    fn prepend_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(stride == mapping.len() as isize, "invalid stride");

        Self::new(mapping.shape().prepend_dim(size))
    }

    fn remap<M: Mapping<Shape = S>>(mapping: M) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");

        Self::new(mapping.shape())
    }

    fn remove_dim<M: Mapping>(mapping: M, index: usize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(index == 0, "invalid dimension");

        Self::new(mapping.shape().remove_dim(index))
    }

    fn reorder<M: Mapping<Shape = S::Reverse>>(mapping: M) -> Self {
        assert!(S::RANK < 2 && M::Layout::IS_DENSE, "invalid layout");

        Self::new(mapping.shape().reverse())
    }

    fn reshape<M: Mapping<Layout = Dense>>(mapping: M, new_shape: S) -> Self {
        assert!(new_shape.checked_len() == Some(mapping.len()), "length must not change");

        Self::new(new_shape)
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(index == 0, "invalid dimension");

        Self::new(mapping.shape().resize_dim(index, new_size))
    }
}

impl<S: Shape> StridedMapping<S> {
    /// Creates a new, strided layout mapping with the specified shape and strides.
    pub fn new(shape: S, strides: S::Strides) -> Self {
        Self { shape, strides }
    }
}

impl<S: Shape> Default for StridedMapping<S> {
    fn default() -> Self {
        let mut strides = S::Strides::default();

        if S::RANK > 0 {
            strides[S::RANK - 1] = 1;
        }

        Self::new(S::default(), strides)
    }
}

impl<S: Shape> Mapping for StridedMapping<S> {
    type Shape = S;
    type Layout = Strided;

    fn is_contiguous(self) -> bool {
        let mut stride = 1;

        for i in (0..S::RANK).rev() {
            if self.strides[i] != stride {
                return false;
            }

            stride *= self.dim(i) as isize;
        }

        true
    }

    fn shape(self) -> S {
        self.shape
    }

    fn strides(self) -> S::Strides {
        self.strides
    }

    fn linear_offset(self, index: usize) -> isize {
        debug_assert!(index < self.len(), "index out of bounds");

        let mut dividend = index;
        let mut offset = 0;

        for i in (0..S::RANK).rev() {
            offset += self.strides[i] * (dividend % self.dim(i)) as isize;
            dividend /= self.dim(i);
        }

        offset
    }

    fn prepend_dim<M: Mapping>(mapping: M, size: usize, stride: isize) -> Self {
        assert!(S::RANK == M::Shape::RANK + 1, "invalid rank");

        let mut strides = S::Strides::default();

        strides[0] = stride;
        strides[1..].copy_from_slice(&mapping.strides()[..]);

        Self::new(mapping.shape().prepend_dim(size), strides)
    }

    fn remap<M: Mapping<Shape = S>>(mapping: M) -> Self {
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

    fn reorder<M: Mapping<Shape = S::Reverse>>(mapping: M) -> Self {
        let mut strides = S::Strides::default();

        strides[..].copy_from_slice(&mapping.strides()[..]);
        strides[..].reverse();

        Self::new(mapping.shape().reverse(), strides)
    }

    fn reshape<M: Mapping<Layout = Strided>>(mapping: M, new_shape: S) -> Self {
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

        let mut j = S::RANK;

        for i in (0..M::Shape::RANK).rev() {
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
            while j > 0 {
                if new_len * new_dims[j - 1] > old_len {
                    break;
                }

                j -= 1;

                new_strides[j] = new_stride;

                new_len *= new_dims[j];
                new_stride *= new_dims[j] as isize;
            }
        }

        // Add remaining dimensions.
        while j > 0 {
            j -= 1;

            new_strides[j] = new_stride;

            new_len *= new_dims[j];
            new_stride *= new_dims[j] as isize;
        }

        assert!(new_len == 0 || valid_layout, "memory layout not compatible");

        Self::new(new_shape, new_strides)
    }

    fn resize_dim<M: Mapping>(mapping: M, index: usize, new_size: usize) -> Self {
        let strides = TryFrom::try_from(&mapping.strides()[..]).expect("invalid rank");

        Self::new(mapping.shape().resize_dim(index, new_size), strides)
    }
}
