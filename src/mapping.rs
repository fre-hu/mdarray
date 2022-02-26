use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Range, RangeBounds};
use std::slice::{self, Iter, IterMut};

use crate::dim::{Const, Dim, Shape, U1};
use crate::format::{Dense, Format, General, Linear, Strided};
use crate::iter::{LinearIter, LinearIterMut};
use crate::layout::{DenseLayout, GeneralLayout, Layout, LinearLayout, StridedLayout};
use crate::order::Order;
use crate::span::SpanBase;

pub trait Mapping<D: Dim, F: Format, O: Order>: Copy + Debug + Default {
    fn add_dim(self, size: usize, stride: isize) -> Layout<D::Higher, F, O>;
    fn flatten(self) -> Layout<U1, F::Uniform, O>;
    fn has_linear_indexing(&self) -> bool;
    fn has_slice_indexing(&self) -> bool;
    fn is_contiguous(&self) -> bool;
    fn is_uniformly_strided(&self) -> bool;
    fn iter<T>(span: &SpanBase<T, Layout<D, F, O>>) -> F::Iter<'_, T>;
    fn iter_mut<T>(span: &mut SpanBase<T, Layout<D, F, O>>) -> F::IterMut<'_, T>;
    fn offset(&self, index: D::Shape) -> isize;
    fn remove_dim(self, dim: usize) -> Layout<D::Lower, F, O>;
    fn reshape<S: Shape>(self, shape: S) -> Layout<S::Dim, F, O>;
    fn resize_dim(self, dim: usize, size: usize) -> Layout<D, F, O>;
    fn shape(&self) -> D::Shape;
    fn size(&self, dim: usize) -> usize;
    fn stride(&self, dim: usize) -> isize;
    fn strides(&self) -> D::Strides;
    fn to_non_uniform(self) -> Layout<D, F::NonUniform, O>;
    fn to_non_unit_strided(self) -> Layout<D, F::NonUnitStrided, O>;

    fn dim(&self, index: usize) -> usize {
        O::select(index, D::RANK - 1 - index)
    }

    fn dims(&self, indices: impl RangeBounds<usize>) -> Range<usize> {
        let range = slice::range(indices, ..D::RANK);

        O::select(range.clone(), D::RANK - range.end..D::RANK - range.start)
    }

    fn len(&self) -> usize {
        self.shape().as_ref().iter().product()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DenseMapping<D: Dim, O: Order> {
    shape: D::Shape,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GeneralMapping<D: Dim, O: Order> {
    shape: D::Shape,
    outer_strides: <D::Lower as Dim>::Strides,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearMapping<D: Dim, O: Order> {
    shape: D::Shape,
    inner_stride: <D::MaxOne as Dim>::Strides,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug)]
pub struct StridedMapping<D: Dim, O: Order> {
    shape: D::Shape,
    strides: D::Strides,
    phantom: PhantomData<O>,
}

pub(crate) trait StaticMapping<D: Dim, O: Order> {
    const MAP: DenseMapping<D, O>;
}

impl<D: Dim, O: Order> DenseMapping<D, O> {
    pub fn new(shape: D::Shape) -> Self {
        Self { shape, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Dense, O> for DenseMapping<D, O> {
    fn add_dim(self, size: usize, stride: isize) -> DenseLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");
        assert!(stride == self.len() as isize, "invalid stride");

        let mut shape = <D::Higher as Dim>::Shape::default();

        shape.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(self.shape.as_ref());
        shape.as_mut()[O::select(D::RANK, 0)] = size;

        DenseLayout::new(shape)
    }

    fn flatten(self) -> DenseLayout<U1, O> {
        DenseLayout::new([self.len()])
    }

    fn has_linear_indexing(&self) -> bool {
        true
    }

    fn has_slice_indexing(&self) -> bool {
        true
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn is_uniformly_strided(&self) -> bool {
        true
    }

    fn iter<T>(span: &SpanBase<T, DenseLayout<D, O>>) -> Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, DenseLayout<D, O>>) -> IterMut<'_, T> {
        span.as_mut_slice().iter_mut()
    }

    fn offset(&self, index: D::Shape) -> isize {
        let mut offset = 0;
        let mut stride = 1;

        for i in 0..D::RANK {
            offset += stride * index.as_ref()[self.dim(i)];
            stride *= self.size(self.dim(i));
        }

        offset as isize
    }

    fn remove_dim(self, dim: usize) -> DenseLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == self.dim(D::RANK - 1), "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();

        if D::RANK > 1 {
            shape.as_mut().copy_from_slice(&self.shape.as_ref()[self.dims(..D::RANK - 1)]);
        }

        DenseLayout::new(shape)
    }

    fn reshape<S: Shape>(self, shape: S) -> DenseLayout<S::Dim, O> {
        let len = shape.as_ref().iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(len == self.len(), "array length must not change");

        DenseLayout::new(shape)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> DenseLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == self.dim(D::RANK - 1), "invalid dimension");

        self.shape.as_mut()[dim] = size;

        DenseLayout::new(self.shape)
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        let inner_dims = self.dims(..self.dim(dim));

        self.shape.as_ref()[inner_dims].iter().product::<usize>() as isize
    }

    fn strides(&self) -> D::Strides {
        let mut strides = D::Strides::default();
        let mut stride = 1;

        for i in 0..D::RANK {
            strides.as_mut()[self.dim(i)] = stride as isize;
            stride *= self.size(self.dim(i));
        }

        strides
    }

    fn to_non_uniform(self) -> GeneralLayout<D, O> {
        StridedLayout::new(self.shape, self.strides()).to_general()
    }

    fn to_non_unit_strided(self) -> LinearLayout<D, O> {
        let mut inner_stride = <D::MaxOne as Dim>::Strides::default();

        if D::RANK > 0 {
            inner_stride.as_mut()[0] = 1;
        }

        LinearLayout::new(self.shape, inner_stride)
    }
}

impl<D: Dim, O: Order> GeneralMapping<D, O> {
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        Self { shape, outer_strides, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, General, O> for GeneralMapping<D, O> {
    fn add_dim(self, size: usize, stride: isize) -> GeneralLayout<D::Higher, O> {
        StridedMapping::<D, O>::new(self.shape, self.strides()).add_dim(size, stride).to_general()
    }

    fn flatten(self) -> DenseLayout<U1, O> {
        assert!(self.is_contiguous(), "array layout not contiguous");

        DenseLayout::new([self.len()])
    }

    fn has_linear_indexing(&self) -> bool {
        D::RANK < 2
    }

    fn has_slice_indexing(&self) -> bool {
        D::RANK < 2
    }

    fn is_contiguous(&self) -> bool {
        if D::RANK > 1 {
            let mut stride = self.size(self.dim(0));

            for i in 1..D::RANK {
                if self.outer_strides.as_ref()[self.dim(i) - O::select(1, 0)] != stride as isize {
                    return false;
                }

                stride *= self.size(self.dim(i))
            }
        }

        true
    }

    fn is_uniformly_strided(&self) -> bool {
        self.is_contiguous()
    }

    fn iter<T>(span: &SpanBase<T, GeneralLayout<D, O>>) -> Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, GeneralLayout<D, O>>) -> IterMut<'_, T> {
        span.as_mut_slice().iter_mut()
    }

    fn offset(&self, index: D::Shape) -> isize {
        let mut offset = 0;

        if D::RANK > 0 {
            offset = index.as_ref()[self.dim(0)] as isize;

            for i in 1..D::RANK {
                offset += self.stride(self.dim(i)) * index.as_ref()[self.dim(i)] as isize;
            }
        }

        offset
    }

    fn remove_dim(self, dim: usize) -> GeneralLayout<D::Lower, O> {
        assert!(D::RANK == 1 || dim != self.dim(0), "invalid dimension");

        StridedMapping::<D, O>::new(self.shape, self.strides()).remove_dim(dim).to_general()
    }

    fn reshape<S: Shape>(self, shape: S) -> GeneralLayout<S::Dim, O> {
        StridedMapping::<D, O>::new(self.shape, self.strides()).reshape(shape).to_general()
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> GeneralLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");

        self.shape.as_mut()[dim] = size;

        GeneralLayout::new(self.shape, self.outer_strides)
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        if dim == self.dim(0) { 1 } else { self.outer_strides.as_ref()[dim - O::select(1, 0)] }
    }

    fn strides(&self) -> D::Strides {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            strides.as_mut()[self.dim(0)] = 1;
            strides.as_mut()[self.dims(1..)].copy_from_slice(self.outer_strides.as_ref());
        }

        strides
    }

    fn to_non_uniform(self) -> GeneralLayout<D, O> {
        GeneralLayout::new(self.shape, self.outer_strides)
    }

    fn to_non_unit_strided(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape, self.strides())
    }
}

impl<D: Dim, O: Order> LinearMapping<D, O> {
    pub fn new(shape: D::Shape, inner_stride: <D::MaxOne as Dim>::Strides) -> Self {
        Self { shape, inner_stride, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Linear, O> for LinearMapping<D, O> {
    fn add_dim(self, size: usize, stride: isize) -> LinearLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Higher as Dim>::Shape::default();
        let mut inner_stride = <<D::Higher as Dim>::MaxOne as Dim>::Strides::default();

        shape.as_mut()[O::select(D::RANK, 0)] = size;

        if D::RANK == 0 {
            inner_stride.as_mut()[0] = stride;
        } else {
            inner_stride.as_mut()[0] = self.inner_stride.as_ref()[0];

            assert!(stride == inner_stride.as_ref()[0] * self.len() as isize, "invalid stride");

            shape.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)]
                .copy_from_slice(self.shape.as_ref());
        }

        LinearLayout::new(shape, inner_stride)
    }

    fn flatten(self) -> LinearLayout<U1, O> {
        let inner_stride = if D::RANK > 0 { self.inner_stride.as_ref()[0] } else { 1 };

        LinearLayout::new([self.len()], [inner_stride])
    }

    fn has_linear_indexing(&self) -> bool {
        true
    }

    fn has_slice_indexing(&self) -> bool {
        D::RANK == 0
    }

    fn is_contiguous(&self) -> bool {
        D::RANK == 0 || self.inner_stride.as_ref()[0] == 1
    }

    fn is_uniformly_strided(&self) -> bool {
        true
    }

    fn iter<T>(span: &SpanBase<T, LinearLayout<D, O>>) -> LinearIter<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIter::new_unchecked(span.as_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, LinearLayout<D, O>>) -> LinearIterMut<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIterMut::new_unchecked(span.as_mut_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn offset(&self, index: D::Shape) -> isize {
        let mut offset = 0;

        if D::RANK > 0 {
            let mut stride = self.inner_stride.as_ref()[0];

            for i in 0..D::RANK {
                offset += stride * index.as_ref()[self.dim(i)] as isize;
                stride *= self.size(self.dim(i)) as isize;
            }
        }

        offset
    }

    fn remove_dim(self, dim: usize) -> LinearLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == 0 || dim == D::RANK - 1, "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();
        let mut inner_stride = <<D::Lower as Dim>::MaxOne as Dim>::Strides::default();

        if D::RANK > 1 {
            shape.as_mut()[..dim].copy_from_slice(&self.shape.as_ref()[..dim]);
            shape.as_mut()[dim..].copy_from_slice(&self.shape.as_ref()[dim + 1..]);

            let size = if dim == self.dim(0) { self.size(dim) } else { 1 };

            inner_stride.as_mut()[0] = self.inner_stride.as_ref()[0] * size as isize;
        }

        LinearLayout::new(shape, inner_stride)
    }

    fn reshape<S: Shape>(self, shape: S) -> LinearLayout<S::Dim, O> {
        let len = shape.as_ref().iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(len == self.len(), "array length must not change");

        let mut inner_stride = <<S::Dim as Dim>::MaxOne as Dim>::Strides::default();

        if S::Dim::RANK > 0 {
            inner_stride.as_mut()[0] = if D::RANK > 0 { self.inner_stride.as_ref()[0] } else { 1 };
        }

        LinearLayout::new(shape, inner_stride)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> LinearLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == self.dim(D::RANK - 1), "invalid dimension");

        self.shape.as_mut()[dim] = size;

        LinearLayout::new(self.shape, self.inner_stride)
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        let inner_dims = self.dims(..self.dim(dim));
        let inner_stride = self.inner_stride.as_ref()[0];

        self.shape.as_ref()[inner_dims].iter().fold(inner_stride, |acc, &x| acc * x as isize)
    }

    fn strides(&self) -> D::Strides {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            let mut stride = self.inner_stride.as_ref()[0];

            for i in 0..D::RANK {
                strides.as_mut()[self.dim(i)] = stride;
                stride *= self.size(self.dim(i)) as isize;
            }
        }

        strides
    }

    fn to_non_uniform(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape, self.strides())
    }

    fn to_non_unit_strided(self) -> LinearLayout<D, O> {
        LinearLayout::new(self.shape, self.inner_stride)
    }
}

impl<D: Dim, O: Order> Default for LinearMapping<D, O> {
    fn default() -> Self {
        let mut inner_stride = <D::MaxOne as Dim>::Strides::default();

        if D::RANK > 0 {
            inner_stride.as_mut()[0] = 1;
        }

        Self { shape: D::Shape::default(), inner_stride, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> StridedMapping<D, O> {
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { shape, strides, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Strided, O> for StridedMapping<D, O> {
    fn add_dim(self, size: usize, stride: isize) -> StridedLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Higher as Dim>::Shape::default();
        let mut strides = <D::Higher as Dim>::Strides::default();

        shape.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(self.shape.as_ref());
        shape.as_mut()[O::select(D::RANK, 0)] = size;

        strides.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)]
            .copy_from_slice(self.strides.as_ref());
        strides.as_mut()[O::select(D::RANK, 0)] = stride;

        StridedLayout::new(shape, strides)
    }

    fn flatten(self) -> LinearLayout<U1, O> {
        assert!(self.is_uniformly_strided(), "array layout not uniformly strided");

        let inner_stride = if D::RANK > 0 { self.stride(self.dim(0)) } else { 1 };

        LinearLayout::new([self.len()], [inner_stride])
    }

    fn has_linear_indexing(&self) -> bool {
        D::RANK < 2
    }

    fn has_slice_indexing(&self) -> bool {
        D::RANK == 0
    }

    fn is_contiguous(&self) -> bool {
        let mut stride = 1;

        for i in 0..D::RANK {
            if self.stride(self.dim(i)) != stride as isize {
                return false;
            }

            stride *= self.size(self.dim(i))
        }

        true
    }

    fn is_uniformly_strided(&self) -> bool {
        if D::RANK > 1 {
            let mut stride = self.stride(self.dim(0));

            for i in 1..D::RANK {
                stride *= self.size(self.dim(i - 1)) as isize;

                if self.stride(self.dim(i)) != stride {
                    return false;
                }
            }
        }

        true
    }

    fn iter<T>(span: &SpanBase<T, StridedLayout<D, O>>) -> LinearIter<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIter::new_unchecked(span.as_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, StridedLayout<D, O>>) -> LinearIterMut<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIterMut::new_unchecked(span.as_mut_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn offset(&self, index: D::Shape) -> isize {
        let mut offset = 0;

        for i in 0..D::RANK {
            offset += self.stride(i) * index.as_ref()[i] as isize;
        }

        offset
    }

    fn remove_dim(self, dim: usize) -> StridedLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");

        let mut shape = <D::Lower as Dim>::Shape::default();
        let mut strides = <D::Lower as Dim>::Strides::default();

        if D::RANK > 1 {
            shape.as_mut()[..dim].copy_from_slice(&self.shape.as_ref()[..dim]);
            shape.as_mut()[dim..].copy_from_slice(&self.shape.as_ref()[dim + 1..]);

            strides.as_mut()[..dim].copy_from_slice(&self.strides.as_ref()[..dim]);
            strides.as_mut()[dim..].copy_from_slice(&self.strides.as_ref()[dim + 1..]);
        }

        StridedLayout::new(shape, strides)
    }

    fn reshape<S: Shape>(self, shape: S) -> StridedLayout<S::Dim, O> {
        let mut strides = <S::Dim as Dim>::Strides::default();

        let mut old_len = 1usize;
        let mut new_len = 1usize;

        let mut old_stride = 1;
        let mut new_stride = 1;

        let mut k = 0;

        for i in 0..D::RANK {
            // Add dimensions within a contiguous region.
            while k < S::Dim::RANK {
                let dim = O::select(k, S::Dim::RANK - 1 - k);
                let len = new_len.saturating_mul(shape.as_ref()[dim]);

                if len > old_len {
                    break;
                }

                strides.as_mut()[dim] = new_stride;

                new_len = len;
                new_stride *= shape.as_ref()[dim] as isize;

                k += 1;
            }

            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = self.stride(self.dim(i));
                new_stride = old_stride;
            } else {
                assert!(old_stride == self.stride(self.dim(i)), "memory layout not compatible");
            }

            let size = self.size(self.dim(i));

            old_len *= size;
            old_stride *= size as isize;
        }

        // Add remaining dimensions.
        while k < S::Dim::RANK {
            let dim = O::select(k, S::Dim::RANK - 1 - k);

            strides.as_mut()[dim] = new_stride;

            new_len = new_len.saturating_mul(shape.as_ref()[dim]);
            new_stride *= shape.as_ref()[dim] as isize;

            k += 1;
        }

        assert!(new_len == old_len, "array length must not change");

        StridedLayout::new(shape, strides)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> StridedLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");

        self.shape.as_mut()[dim] = size;

        StridedLayout::new(self.shape, self.strides)
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        self.strides.as_ref()[dim]
    }

    fn strides(&self) -> D::Strides {
        self.strides
    }

    fn to_non_uniform(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape, self.strides)
    }

    fn to_non_unit_strided(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape, self.strides)
    }
}

impl<D: Dim, O: Order> Default for StridedMapping<D, O> {
    fn default() -> Self {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            strides.as_mut()[O::select(0, D::RANK - 1)] = 1;
        }

        Self { shape: D::Shape::default(), strides, phantom: PhantomData }
    }
}

macro_rules! impl_static_mapping {
    ($n:tt, ($($xyz:tt),+)) => {
        #[allow(unused_parens)]
        impl<O: Order, $(const $xyz: usize),+> StaticMapping<Const<$n>, O> for ($(Const<$xyz>),+) {
            const MAP: DenseMapping<Const<$n>, O> =
                DenseMapping { shape: [$($xyz),+], phantom: PhantomData };
        }
    };
}

impl_static_mapping!(1, (X));
impl_static_mapping!(2, (X, Y));
impl_static_mapping!(3, (X, Y, Z));
impl_static_mapping!(4, (X, Y, Z, W));
impl_static_mapping!(5, (X, Y, Z, W, U));
impl_static_mapping!(6, (X, Y, Z, W, U, V));
