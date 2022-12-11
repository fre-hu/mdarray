use std::fmt::{Debug, Formatter, Result};

use crate::dim::{Dim, Rank, Shape};
use crate::format::{Dense, Flat, Format, General, Strided};
use crate::iter::{LinearIter, LinearIterMut};
use crate::layout::{DenseLayout, FlatLayout, GeneralLayout, Layout, StridedLayout};
use crate::order::Order;
use crate::span::SpanBase;

pub trait Mapping: Copy + Debug + Default {
    type Dim: Dim;
    type Format: Format;

    fn is_contiguous(self) -> bool;
    fn is_uniformly_strided(self) -> bool;
    fn shape(self) -> <Self::Dim as Dim>::Shape;
    fn strides(self) -> <Self::Dim as Dim>::Strides;

    fn add_dim<F: Format>(layout: Reformat<Self, F>, size: usize, stride: isize) -> Higher<Self>;
    fn flatten(self) -> Flatten<Self>;
    fn reformat<F: Format>(layout: Reformat<Self, F>) -> Layout<Self::Dim, Self::Format>;
    fn remove_dim<F: Format>(layout: Reformat<Self, F>, dim: usize) -> Lower<Self>;
    fn reshape<S: Shape, F: Format>(layout: Reformat<Self, F>, new_shape: S) -> Reshape<Self, S>;
    fn resize_dim(self, dim: usize, new_size: usize) -> Layout<Self::Dim, Self::Format>;

    fn iter<T>(span: &SpanBase<T, Self::Dim, Self::Format>) -> Iter<'_, T, Self>;
    fn iter_mut<T>(span: &mut SpanBase<T, Self::Dim, Self::Format>) -> IterMut<'_, T, Self>;

    fn len(self) -> usize {
        self.shape()[..].iter().product()
    }
}

pub struct DenseMapping<D: Dim> {
    shape: D::Shape,
}

pub struct FlatMapping<D: Dim> {
    shape: D::Shape,
    inner_stride: isize,
}

pub struct GeneralMapping<D: Dim> {
    shape: D::Shape,
    outer_strides: <D::Lower as Dim>::Strides,
}

pub struct StridedMapping<D: Dim> {
    shape: D::Shape,
    strides: D::Strides,
}

type Flatten<M> = Layout<Rank<1, <<M as Mapping>::Dim as Dim>::Order>, Uniform<M>>;
type Higher<M> = Redim<M, <<M as Mapping>::Dim as Dim>::Higher>;
type Iter<'a, T, M> = <<M as Mapping>::Format as Format>::Iter<'a, T>;
type IterMut<'a, T, M> = <<M as Mapping>::Format as Format>::IterMut<'a, T>;
type Lower<M> = Redim<M, <<M as Mapping>::Dim as Dim>::Lower>;
type Redim<M, D> = Layout<D, <M as Mapping>::Format>;
type Reformat<M, F> = Layout<<M as Mapping>::Dim, F>;
type Reshape<M, S> = Redim<M, <S as Shape>::Dim<<<M as Mapping>::Dim as Dim>::Order>>;
type Uniform<M> = <<M as Mapping>::Format as Format>::Uniform;

impl<D: Dim> DenseMapping<D> {
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
        f.debug_struct("DenseLayout").field("shape", &self.shape).finish()
    }
}

impl<D: Dim> Default for DenseMapping<D> {
    fn default() -> Self {
        Self { shape: Default::default() }
    }
}

impl<D: Dim> Mapping for DenseMapping<D> {
    type Dim = D;
    type Format = Dense;

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
            strides[D::dim(i)] = stride as isize;
            stride *= self.shape[D::dim(i)];
        }

        strides
    }

    fn add_dim<F: Format>(layout: Reformat<Self, F>, size: usize, stride: isize) -> Higher<Self> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");
        assert!(F::IS_UNIFORM && F::IS_UNIT_STRIDED, "invalid format");
        assert!(stride == layout.len() as isize, "invalid stride");

        let mut shape = <D::Higher as Dim>::Shape::default();

        shape[D::Higher::dims(..D::RANK)].copy_from_slice(&layout.shape()[..]);
        shape[D::Higher::dim(D::RANK)] = size;

        DenseLayout::new(shape)
    }

    fn flatten(self) -> Flatten<Self> {
        DenseLayout::new([self.len()])
    }

    fn reformat<F: Format>(layout: Reformat<Self, F>) -> Layout<Self::Dim, Self::Format> {
        assert!(layout.is_contiguous(), "array layout not contiguous");

        DenseLayout::new(layout.shape())
    }

    fn remove_dim<F: Format>(layout: Reformat<Self, F>, dim: usize) -> Lower<Self> {
        assert!(D::RANK < 2 || F::IS_UNIT_STRIDED, "invalid format");
        assert!(D::RANK < 3 || F::IS_UNIFORM, "invalid format");
        assert!(D::RANK > 0 && dim == D::dim(D::RANK - 1), "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();

        if D::RANK > 1 {
            shape[..].copy_from_slice(&layout.shape()[D::dims(..D::RANK - 1)]);
        }

        DenseLayout::new(shape)
    }

    fn reshape<S: Shape, F: Format>(layout: Reformat<Self, F>, new_shape: S) -> Reshape<Self, S> {
        assert!(F::IS_UNIFORM && F::IS_UNIT_STRIDED, "invalid format");

        let new_len = new_shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(new_len == layout.len(), "array length must not change");

        DenseLayout::new(new_shape)
    }

    fn resize_dim(mut self, dim: usize, new_size: usize) -> Layout<Self::Dim, Self::Format> {
        assert!(D::RANK > 0 && dim == D::dim(D::RANK - 1), "invalid dimension");

        self.shape[dim] = new_size;

        DenseLayout::new(self.shape)
    }

    fn iter<T>(span: &SpanBase<T, Self::Dim, Self::Format>) -> Iter<'_, T, Self> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, Self::Dim, Self::Format>) -> IterMut<'_, T, Self> {
        span.as_mut_slice().iter_mut()
    }
}

impl<D: Dim> FlatMapping<D> {
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
        f.debug_struct("FlatLayout")
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
    type Format = Flat;

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
            strides[D::dim(i)] = stride;
            stride *= self.shape[D::dim(i)] as isize;
        }

        strides
    }

    fn add_dim<F: Format>(layout: Reformat<Self, F>, size: usize, stride: isize) -> Higher<Self> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");
        assert!(F::IS_UNIFORM, "invalid format");

        let inner_stride = if D::RANK > 0 { layout.stride(D::dim(0)) } else { stride };

        assert!(stride == inner_stride * layout.len() as isize, "invalid stride");

        let mut shape = <D::Higher as Dim>::Shape::default();

        shape[D::Higher::dims(..D::RANK)].copy_from_slice(&layout.shape()[..]);
        shape[D::Higher::dim(D::RANK)] = size;

        FlatLayout::new(shape, inner_stride)
    }

    fn flatten(self) -> Flatten<Self> {
        FlatLayout::new([self.len()], self.inner_stride)
    }

    fn reformat<F: Format>(layout: Reformat<Self, F>) -> Layout<Self::Dim, Self::Format> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(layout.is_uniformly_strided(), "array layout not uniformly strided");

        FlatLayout::new(layout.shape(), layout.stride(D::dim(0)))
    }

    fn remove_dim<F: Format>(layout: Reformat<Self, F>, dim: usize) -> Lower<Self> {
        assert!(D::RANK > 1, "invalid rank");
        assert!(D::RANK < 3 || F::IS_UNIFORM, "invalid format");
        assert!(dim == 0 || dim == D::RANK - 1, "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();

        shape[..dim].copy_from_slice(&layout.shape()[..dim]);
        shape[dim..].copy_from_slice(&layout.shape()[dim + 1..]);

        let size = if dim == D::dim(0) { layout.size(dim) } else { 1 };

        FlatLayout::new(shape, layout.stride(D::dim(0)) * size as isize)
    }

    fn reshape<S: Shape, F: Format>(layout: Reformat<Self, F>, new_shape: S) -> Reshape<Self, S> {
        assert!(<S::Dim<D::Order> as Dim>::RANK > 0, "invalid rank");
        assert!(F::IS_UNIFORM, "invalid format");

        let new_len = new_shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(new_len == layout.len(), "array length must not change");

        FlatLayout::new(new_shape, if D::RANK > 0 { layout.stride(D::dim(0)) } else { 1 })
    }

    fn resize_dim(mut self, dim: usize, new_size: usize) -> Layout<Self::Dim, Self::Format> {
        assert!(dim == D::dim(D::RANK - 1), "invalid dimension");

        self.shape[dim] = new_size;

        FlatLayout::new(self.shape, self.inner_stride)
    }

    fn iter<T>(span: &SpanBase<T, Self::Dim, Self::Format>) -> Iter<'_, T, Self> {
        let layout = span.layout().flatten();

        unsafe { LinearIter::new_unchecked(span.as_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, Self::Dim, Self::Format>) -> IterMut<'_, T, Self> {
        let layout = span.layout().flatten();

        unsafe { LinearIterMut::new_unchecked(span.as_mut_ptr(), layout.size(0), layout.stride(0)) }
    }
}

impl<D: Dim> GeneralMapping<D> {
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
        f.debug_struct("GeneralLayout")
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
    type Format = General;

    fn is_contiguous(self) -> bool {
        let mut stride = self.shape[D::dim(0)];

        for i in 1..D::RANK {
            if self.outer_strides[D::dim(i) - D::Order::select(1, 0)] != stride as isize {
                return false;
            }

            stride *= self.shape[D::dim(i)];
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

        strides[D::dim(0)] = 1;
        strides[D::dims(1..)].copy_from_slice(&self.outer_strides[..]);

        strides
    }

    fn add_dim<F: Format>(layout: Reformat<Self, F>, size: usize, stride: isize) -> Higher<Self> {
        assert!(D::RANK > 0 && D::Higher::RANK > D::RANK, "invalid rank");
        assert!(F::IS_UNIT_STRIDED, "invalid format");

        StridedMapping::add_dim(layout, size, stride).reformat()
    }

    fn flatten(self) -> Flatten<Self> {
        assert!(self.is_contiguous(), "array layout not contiguous");

        DenseLayout::new([self.len()])
    }

    fn reformat<F: Format>(layout: Reformat<Self, F>) -> Layout<Self::Dim, Self::Format> {
        assert!(D::RANK > 1, "invalid rank");
        assert!(layout.stride(D::dim(0)) == 1, "inner stride not unitary");

        let mut outer_strides = <D::Lower as Dim>::Strides::default();

        outer_strides[..].copy_from_slice(&layout.strides()[D::dims(1..)]);

        GeneralLayout::new(layout.shape(), outer_strides)
    }

    fn remove_dim<F: Format>(layout: Reformat<Self, F>, dim: usize) -> Lower<Self> {
        assert!(D::RANK > 2, "invalid rank");
        assert!(F::IS_UNIT_STRIDED, "invalid format");
        assert!(dim != D::dim(0), "invalid dimension");

        StridedMapping::remove_dim(layout, dim).reformat()
    }

    fn reshape<S: Shape, F: Format>(layout: Reformat<Self, F>, new_shape: S) -> Reshape<Self, S> {
        assert!(<S::Dim<D::Order> as Dim>::RANK > 1, "invalid rank");
        assert!(F::IS_UNIT_STRIDED, "invalid format");

        StridedMapping::reshape(layout, new_shape).reformat()
    }

    fn resize_dim(mut self, dim: usize, new_size: usize) -> Layout<Self::Dim, Self::Format> {
        assert!(dim < D::RANK, "invalid dimension");

        self.shape[dim] = new_size;

        GeneralLayout::new(self.shape, self.outer_strides)
    }

    fn iter<T>(_: &SpanBase<T, Self::Dim, Self::Format>) -> Iter<'_, T, Self> {
        panic!("invalid format");
    }

    fn iter_mut<T>(_: &mut SpanBase<T, Self::Dim, Self::Format>) -> IterMut<'_, T, Self> {
        panic!("invalid format");
    }
}

impl<D: Dim> StridedMapping<D> {
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
        f.debug_struct("StridedLayout")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

impl<D: Dim> Default for StridedMapping<D> {
    fn default() -> Self {
        assert!(D::RANK > 1, "invalid rank");

        let mut strides = D::Strides::default();

        strides[D::dim(0)] = 1;

        Self { shape: Default::default(), strides }
    }
}

impl<D: Dim> Mapping for StridedMapping<D> {
    type Dim = D;
    type Format = Strided;

    fn is_contiguous(self) -> bool {
        self.strides[D::dim(0)] == 1 && self.is_uniformly_strided()
    }

    fn is_uniformly_strided(self) -> bool {
        let mut stride = self.strides[D::dim(0)];

        for i in 1..D::RANK {
            stride *= self.shape[D::dim(i - 1)] as isize;

            if self.strides[D::dim(i)] != stride {
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

    fn add_dim<F: Format>(layout: Reformat<Self, F>, size: usize, stride: isize) -> Higher<Self> {
        assert!(D::RANK > 0 && D::Higher::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Higher as Dim>::Shape::default();
        let mut strides = <D::Higher as Dim>::Strides::default();

        shape[D::Higher::dims(..D::RANK)].copy_from_slice(&layout.shape()[..]);
        shape[D::Higher::dim(D::RANK)] = size;

        strides[D::Higher::dims(..D::RANK)].copy_from_slice(&layout.strides()[..]);
        strides[D::Higher::dim(D::RANK)] = stride;

        StridedLayout::new(shape, strides)
    }

    fn flatten(self) -> Flatten<Self> {
        assert!(self.is_uniformly_strided(), "array layout not uniformly strided");

        FlatLayout::new([self.len()], self.strides[D::dim(0)])
    }

    fn reformat<F: Format>(layout: Reformat<Self, F>) -> Layout<Self::Dim, Self::Format> {
        assert!(D::RANK > 1, "invalid rank");

        StridedLayout::new(layout.shape(), layout.strides())
    }

    fn remove_dim<F: Format>(layout: Reformat<Self, F>, dim: usize) -> Lower<Self> {
        assert!(D::RANK > 2, "invalid rank");
        assert!(dim < D::RANK, "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();
        let mut strides = <D::Lower as Dim>::Strides::default();

        shape[..dim].copy_from_slice(&layout.shape()[..dim]);
        shape[dim..].copy_from_slice(&layout.shape()[dim + 1..]);

        strides[..dim].copy_from_slice(&layout.strides()[..dim]);
        strides[dim..].copy_from_slice(&layout.strides()[dim + 1..]);

        StridedLayout::new(shape, strides)
    }

    fn reshape<S: Shape, F: Format>(layout: Reformat<Self, F>, new_shape: S) -> Reshape<Self, S> {
        assert!(<S::Dim<D::Order> as Dim>::RANK > 1, "invalid rank");

        let old_shape = layout.shape();
        let old_strides = layout.strides();

        let mut new_strides = <S::Dim<D::Order> as Dim>::Strides::default();

        let mut old_len = 1usize;
        let mut new_len = 1usize;

        let mut old_stride = 1;
        let mut new_stride = 1;

        let mut k = 0;

        for i in 0..D::RANK {
            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = old_strides[D::dim(i)];
                new_stride = old_stride;
            } else {
                assert!(old_stride == old_strides[D::dim(i)], "memory layout not compatible");
            }

            old_len *= old_shape[D::dim(i)];
            old_stride *= old_shape[D::dim(i)] as isize;

            // Add dimensions within the current region.
            while k < <S::Dim<D::Order> as Dim>::RANK {
                let dim = <S::Dim<D::Order> as Dim>::dim(k);
                let len = new_len.saturating_mul(new_shape[dim]);

                if len > old_len {
                    break;
                }

                new_strides[dim] = new_stride;

                new_len = len;
                new_stride *= new_shape[dim] as isize;

                k += 1;
            }
        }

        // Add remaining dimensions.
        while k < <S::Dim<D::Order> as Dim>::RANK {
            let dim = <S::Dim<D::Order> as Dim>::dim(k);

            new_strides[dim] = new_stride;

            new_len = new_len.saturating_mul(new_shape[dim]);
            new_stride *= new_shape[dim] as isize;

            k += 1;
        }

        assert!(new_len == old_len, "array length must not change");

        StridedLayout::new(new_shape, new_strides)
    }

    fn resize_dim(mut self, dim: usize, new_size: usize) -> Layout<Self::Dim, Self::Format> {
        assert!(dim < D::RANK, "invalid dimension");

        self.shape[dim] = new_size;

        StridedLayout::new(self.shape, self.strides)
    }

    fn iter<T>(_: &SpanBase<T, Self::Dim, Self::Format>) -> Iter<'_, T, Self> {
        panic!("invalid format");
    }

    fn iter_mut<T>(_: &mut SpanBase<T, Self::Dim, Self::Format>) -> IterMut<'_, T, Self> {
        panic!("invalid format");
    }
}
