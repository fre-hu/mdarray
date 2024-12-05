use std::fmt::Debug;
use std::hash::Hash;

use crate::dim::Dims;
use crate::layout::{Dense, Layout, Strided};
use crate::shape::{DynRank, Shape};

/// Array layout mapping trait, including shape and strides.
pub trait Mapping: Clone + Debug + Default + Eq + Hash + Send + Sync {
    /// Array shape type.
    type Shape: Shape;

    /// Array layout type.
    type Layout: Layout<Mapping<Self::Shape> = Self>;

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    fn is_contiguous(&self) -> bool;

    /// Returns the array shape.
    fn shape(&self) -> &Self::Shape;

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn stride(&self, index: usize) -> isize;

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(&self, index: usize) -> usize {
        self.shape().dim(index)
    }

    /// Returns the number of elements in each dimension.
    fn dims(&self) -> &[usize]
    where
        Self: Mapping<Shape = DynRank>,
    {
        self.shape().dims()
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.shape().is_empty()
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.shape().len()
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        self.shape().rank()
    }

    #[doc(hidden)]
    fn for_each_stride<F: FnMut(usize, isize)>(&self, f: F);

    #[doc(hidden)]
    fn inner_stride(&self) -> isize;

    #[doc(hidden)]
    fn linear_offset(&self, index: usize) -> isize;

    #[doc(hidden)]
    fn permute<M: Mapping>(mapping: &M, perm: &[usize]) -> Self;

    #[doc(hidden)]
    fn prepend_dim<M: Mapping>(mapping: &M, size: usize, stride: isize) -> Self;

    #[doc(hidden)]
    fn remap<M: Mapping>(mapping: &M) -> Self;

    #[doc(hidden)]
    fn remove_dim<M: Mapping>(mapping: &M, index: usize) -> Self;

    #[doc(hidden)]
    fn reorder<M: Mapping<Shape = <Self::Shape as Shape>::Reverse>>(mapping: &M) -> Self;

    #[doc(hidden)]
    fn reshape<S: Shape>(&self, new_shape: S) -> <Self::Layout as Layout>::Mapping<S>;

    #[doc(hidden)]
    fn resize_dim<M: Mapping>(mapping: &M, index: usize, new_size: usize) -> Self;

    #[doc(hidden)]
    fn shape_mut(&mut self) -> &mut Self::Shape;

    #[doc(hidden)]
    fn offset(&self, index: &[usize]) -> isize {
        debug_assert!(index.len() == self.rank(), "invalid rank");

        let mut offset = 0;

        self.for_each_stride(|i, stride| {
            debug_assert!(index[i] < self.dim(i), "index out of bounds");

            offset += stride * index[i] as isize;
        });

        offset
    }
}

/// Dense layout mapping type.
#[derive(Debug, Default, Eq, Hash, PartialEq)]
pub struct DenseMapping<S: Shape> {
    shape: S,
}

/// Strided layout mapping type.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct StridedMapping<S: Shape> {
    shape: S,
    strides: S::Dims<isize>,
}

impl<S: Shape> DenseMapping<S> {
    /// Creates a new, dense layout mapping with the specified shape.
    pub fn new(shape: S) -> Self {
        Self { shape }
    }
}

impl<S: Shape> Clone for DenseMapping<S> {
    fn clone(&self) -> Self {
        Self::new(self.shape.clone())
    }

    fn clone_from(&mut self, source: &Self) {
        self.shape.clone_from(&source.shape);
    }
}

impl<S: Shape + Copy> Copy for DenseMapping<S> {}

impl<S: Shape> Mapping for DenseMapping<S> {
    type Shape = S;
    type Layout = Dense;

    fn is_contiguous(&self) -> bool {
        true
    }

    fn shape(&self) -> &S {
        &self.shape
    }

    fn stride(&self, index: usize) -> isize {
        assert!(index < self.rank(), "invalid dimension");

        let mut stride = 1;

        for i in index + 1..self.rank() {
            stride *= self.dim(i);
        }

        stride as isize
    }

    fn for_each_stride<F: FnMut(usize, isize)>(&self, mut f: F) {
        let mut stride = 1;

        for i in (0..self.rank()).rev() {
            f(i, stride as isize);
            stride *= self.dim(i);
        }
    }

    fn inner_stride(&self) -> isize {
        // The inner stride should be a compile time constant with dense layout.
        // For static rank 0, we set it to 0 to allow inner rank >0 in iteration.
        if S::RANK == Some(0) {
            0
        } else {
            1
        }
    }

    fn linear_offset(&self, index: usize) -> isize {
        debug_assert!(index < self.len(), "index out of bounds");

        index as isize
    }

    fn permute<M: Mapping>(mapping: &M, perm: &[usize]) -> Self {
        assert!(perm.len() == mapping.rank(), "invalid permutation");

        for i in 0..mapping.rank() {
            assert!(perm[i] == i, "invalid permutation");
        }

        Self::remap(mapping)
    }

    fn prepend_dim<M: Mapping>(mapping: &M, size: usize, stride: isize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(stride == mapping.len() as isize, "invalid stride");

        Self::new(mapping.shape().prepend_dim(size))
    }

    fn remap<M: Mapping>(mapping: &M) -> Self {
        assert!(mapping.is_contiguous(), "mapping not contiguous");

        Self::new(mapping.shape().with_dims(Shape::from_dims))
    }

    fn remove_dim<M: Mapping>(mapping: &M, index: usize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(index == 0, "invalid dimension");

        Self::new(mapping.shape().remove_dim(index))
    }

    fn reorder<M: Mapping<Shape = S::Reverse>>(mapping: &M) -> Self {
        assert!(mapping.rank() < 2 && M::Layout::IS_DENSE, "invalid layout");

        Self::new(mapping.shape().reverse())
    }

    fn reshape<T: Shape>(&self, new_shape: T) -> DenseMapping<T> {
        DenseMapping::new(self.shape.reshape(new_shape))
    }

    fn resize_dim<M: Mapping>(mapping: &M, index: usize, new_size: usize) -> Self {
        assert!(M::Layout::IS_DENSE, "invalid layout");
        assert!(index == 0, "invalid dimension");

        Self::new(mapping.shape().resize_dim(index, new_size))
    }

    fn shape_mut(&mut self) -> &mut S {
        &mut self.shape
    }
}

impl<S: Shape> StridedMapping<S> {
    /// Creates a new, strided layout mapping with the specified shape and strides.
    pub fn new(shape: S, strides: &[isize]) -> Self {
        assert!(shape.rank() == strides.len(), "length mismatch");

        Self { shape, strides: TryFrom::try_from(strides).expect("invalid rank") }
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> &[isize] {
        self.strides.as_ref()
    }
}

impl<S: Shape> Default for StridedMapping<S> {
    fn default() -> Self {
        Self { shape: S::default(), strides: S::Dims::new(S::default().rank()) }
    }
}

impl<S: Shape> Clone for StridedMapping<S> {
    fn clone(&self) -> Self {
        Self { shape: self.shape.clone(), strides: self.strides.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.shape.clone_from(&source.shape);
        self.strides.clone_from(&source.strides);
    }
}

impl<S: Shape<Dims<isize>: Copy> + Copy> Copy for StridedMapping<S> {}

impl<S: Shape> Mapping for StridedMapping<S> {
    type Shape = S;
    type Layout = Strided;

    fn is_contiguous(&self) -> bool {
        let mut stride = 1;

        for i in (0..self.rank()).rev() {
            if self.strides.as_ref()[i] != stride {
                return false;
            }

            stride *= self.dim(i) as isize;
        }

        true
    }

    fn shape(&self) -> &S {
        &self.shape
    }

    fn stride(&self, index: usize) -> isize {
        assert!(index < self.rank(), "invalid dimension");

        self.strides.as_ref()[index]
    }

    fn for_each_stride<F: FnMut(usize, isize)>(&self, mut f: F) {
        for i in 0..self.rank() {
            f(i, self.strides.as_ref()[i])
        }
    }

    fn inner_stride(&self) -> isize {
        if self.rank() > 0 {
            self.strides.as_ref()[self.rank() - 1]
        } else {
            0
        }
    }

    fn linear_offset(&self, index: usize) -> isize {
        debug_assert!(index < self.len(), "index out of bounds");

        let mut dividend = index;
        let mut offset = 0;

        for i in (0..self.rank()).rev() {
            offset += self.strides.as_ref()[i] * (dividend % self.dim(i)) as isize;
            dividend /= self.dim(i);
        }

        offset
    }

    fn permute<M: Mapping>(mapping: &M, perm: &[usize]) -> Self {
        assert!(perm.len() == mapping.rank(), "invalid permutation");

        let mut index_mask = 0;

        for i in 0..mapping.rank() {
            assert!(perm[i] < mapping.rank(), "invalid permutation");

            index_mask |= 1 << perm[i];
        }

        assert!(index_mask == !(usize::MAX << mapping.rank()), "invalid permutation");

        let mut shape = S::new(mapping.rank());
        let mut strides = S::Dims::new(mapping.rank());

        shape.with_mut_dims(|dims| {
            // Calculate inverse permutation
            for i in 0..mapping.rank() {
                dims[perm[i]] = i;
            }

            // Permute strides
            mapping.for_each_stride(|i, stride| strides.as_mut()[dims[i]] = stride);

            // Permute shape
            for i in 0..mapping.rank() {
                dims[i] = mapping.dim(perm[i]);
            }
        });

        Self { shape, strides }
    }

    fn prepend_dim<M: Mapping>(mapping: &M, size: usize, stride: isize) -> Self {
        let mut strides = S::Dims::new(mapping.rank() + 1);

        strides.as_mut()[0] = stride;
        mapping.for_each_stride(|i, stride| strides.as_mut()[i + 1] = stride);

        Self { shape: mapping.shape().prepend_dim(size), strides }
    }

    fn remap<M: Mapping>(mapping: &M) -> Self {
        let mut strides = S::Dims::new(mapping.rank());

        mapping.for_each_stride(|i, stride| strides.as_mut()[i] = stride);

        Self { shape: mapping.shape().with_dims(Shape::from_dims), strides }
    }

    fn remove_dim<M: Mapping>(mapping: &M, index: usize) -> Self {
        assert!(index < mapping.rank(), "invalid dimension");

        let mut strides = S::Dims::new(mapping.rank() - 1);

        mapping.for_each_stride(|i, stride| {
            if i < index {
                strides.as_mut()[i] = stride;
            } else if i > index {
                strides.as_mut()[i - 1] = stride;
            }
        });

        Self { shape: mapping.shape().remove_dim(index), strides }
    }

    fn reorder<M: Mapping<Shape = S::Reverse>>(mapping: &M) -> Self {
        let mut strides = S::Dims::new(mapping.rank());

        mapping.for_each_stride(|i, stride| strides.as_mut()[mapping.rank() - 1 - i] = stride);

        Self { shape: mapping.shape().reverse(), strides }
    }

    fn reshape<T: Shape>(&self, new_shape: T) -> StridedMapping<T> {
        let new_shape = self.shape.reshape(new_shape);
        let mut new_strides = T::Dims::new(new_shape.rank());

        let mut old_len = 1usize;
        let mut new_len = 1usize;

        let mut old_stride = 1;
        let mut new_stride = 1;

        let mut valid_layout = true;

        let mut j = new_shape.rank();

        for i in (0..self.rank()).rev() {
            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = self.strides.as_ref()[i];
                new_stride = old_stride;
            } else {
                valid_layout &= old_stride == self.strides.as_ref()[i];
            }

            old_len *= self.dim(i);
            old_stride *= self.dim(i) as isize;

            // Add dimensions within the current region.
            while j > 0 {
                if new_len * new_shape.dim(j - 1) > old_len {
                    break;
                }

                j -= 1;

                new_strides.as_mut()[j] = new_stride;

                new_len *= new_shape.dim(j);
                new_stride *= new_shape.dim(j) as isize;
            }
        }

        // Add remaining dimensions.
        while j > 0 {
            j -= 1;

            new_strides.as_mut()[j] = new_stride;

            new_len *= new_shape.dim(j);
            new_stride *= new_shape.dim(j) as isize;
        }

        assert!(new_len == 0 || valid_layout, "memory layout not compatible");

        StridedMapping { shape: new_shape, strides: new_strides }
    }

    fn resize_dim<M: Mapping>(mapping: &M, index: usize, new_size: usize) -> Self {
        let mut strides = S::Dims::new(mapping.rank());

        mapping.for_each_stride(|i, stride| strides.as_mut()[i] = stride);

        Self { shape: mapping.shape().resize_dim(index, new_size), strides }
    }

    fn shape_mut(&mut self) -> &mut S {
        &mut self.shape
    }
}
