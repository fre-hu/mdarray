use std::fmt::Debug;
use std::marker::PhantomData;
use std::slice::{Iter, IterMut};

use crate::dim::{Dim, Shape, U1};
use crate::format::{Dense, Flat, Format, General, Strided};
use crate::iter::{LinearIter, LinearIterMut};
use crate::layout::{DenseLayout, FlatLayout, GeneralLayout, Layout, StridedLayout};
use crate::order::Order;
use crate::span::SpanBase;

pub trait Mapping<D: Dim, F: Format, O: Order>: Copy + Debug + Default {
    fn has_linear_indexing(self) -> bool;
    fn has_slice_indexing(self) -> bool;
    fn is_contiguous(self) -> bool;
    fn is_uniformly_strided(self) -> bool;
    fn shape(self) -> D::Shape;
    fn strides(self) -> D::Strides;

    fn add_dim(self, size: usize, stride: isize) -> Layout<D::Higher, F, O>;
    fn flatten(self) -> Layout<U1, F::Uniform, O>;
    fn reformat<G: Format>(layout: Layout<D, G, O>) -> Layout<D, F, O>;
    fn remove_dim(self, dim: usize) -> Layout<D::Lower, F, O>;
    fn reshape<S: Shape>(self, shape: S) -> Layout<S::Dim, F, O>;
    fn resize_dim(self, dim: usize, size: usize) -> Layout<D, F, O>;

    fn iter<T>(span: &SpanBase<T, Layout<D, F, O>>) -> F::Iter<'_, T>;
    fn iter_mut<T>(span: &mut SpanBase<T, Layout<D, F, O>>) -> F::IterMut<'_, T>;

    fn len(self) -> usize {
        self.shape()[..].iter().product()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DenseMapping<D: Dim, O: Order> {
    shape: D::Shape,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug)]
pub struct FlatMapping<D: Dim, O: Order> {
    shape: D::Shape,
    inner_stride: <D::MaxOne as Dim>::Strides,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GeneralMapping<D: Dim, O: Order> {
    shape: D::Shape,
    outer_strides: <D::Lower as Dim>::Strides,
    phantom: PhantomData<O>,
}

#[derive(Clone, Copy, Debug)]
pub struct StridedMapping<D: Dim, O: Order> {
    shape: D::Shape,
    strides: D::Strides,
    phantom: PhantomData<O>,
}

impl<D: Dim, O: Order> DenseMapping<D, O> {
    pub fn new(shape: D::Shape) -> Self {
        Self { shape, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Dense, O> for DenseMapping<D, O> {
    fn has_linear_indexing(self) -> bool {
        true
    }

    fn has_slice_indexing(self) -> bool {
        true
    }

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
            strides[D::dim::<O>(i)] = stride as isize;
            stride *= self.shape[D::dim::<O>(i)];
        }

        strides
    }

    fn add_dim(self, size: usize, stride: isize) -> DenseLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");
        assert!(stride == self.len() as isize, "invalid stride");

        let mut shape = <D::Higher as Dim>::Shape::default();

        shape[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(&self.shape[..]);
        shape[O::select(D::RANK, 0)] = size;

        DenseLayout::new(shape)
    }

    fn flatten(self) -> DenseLayout<U1, O> {
        DenseLayout::new([self.len()])
    }

    fn reformat<F: Format>(layout: Layout<D, F, O>) -> DenseLayout<D, O> {
        assert!(layout.is_contiguous(), "array layout not contiguous");

        DenseLayout::new(layout.shape())
    }

    fn remove_dim(self, dim: usize) -> DenseLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == D::dim::<O>(D::RANK - 1), "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();

        if D::RANK > 1 {
            shape[..].copy_from_slice(&self.shape[D::dims::<O>(..D::RANK - 1)]);
        }

        DenseLayout::new(shape)
    }

    fn reshape<S: Shape>(self, shape: S) -> DenseLayout<S::Dim, O> {
        let len = shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(len == self.len(), "array length must not change");

        DenseLayout::new(shape)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> DenseLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == D::dim::<O>(D::RANK - 1), "invalid dimension");

        self.shape[dim] = size;

        DenseLayout::new(self.shape)
    }

    fn iter<T>(span: &SpanBase<T, DenseLayout<D, O>>) -> Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, DenseLayout<D, O>>) -> IterMut<'_, T> {
        span.as_mut_slice().iter_mut()
    }
}

impl<D: Dim, O: Order> FlatMapping<D, O> {
    pub fn new(shape: D::Shape, inner_stride: <D::MaxOne as Dim>::Strides) -> Self {
        Self { shape, inner_stride, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Default for FlatMapping<D, O> {
    fn default() -> Self {
        let mut inner_stride = <D::MaxOne as Dim>::Strides::default();

        if D::RANK > 0 {
            inner_stride[0] = 1;
        }

        Self { shape: D::Shape::default(), inner_stride, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Flat, O> for FlatMapping<D, O> {
    fn has_linear_indexing(self) -> bool {
        true
    }

    fn has_slice_indexing(self) -> bool {
        D::RANK == 0
    }

    fn is_contiguous(self) -> bool {
        D::RANK == 0 || self.inner_stride[0] == 1
    }

    fn is_uniformly_strided(self) -> bool {
        true
    }

    fn shape(self) -> D::Shape {
        self.shape
    }

    fn strides(self) -> D::Strides {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            let mut stride = self.inner_stride[0];

            for i in 0..D::RANK {
                strides[D::dim::<O>(i)] = stride;
                stride *= self.shape[D::dim::<O>(i)] as isize;
            }
        }

        strides
    }

    fn add_dim(self, size: usize, stride: isize) -> FlatLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Higher as Dim>::Shape::default();
        let mut inner_stride = <<D::Higher as Dim>::MaxOne as Dim>::Strides::default();

        shape[O::select(D::RANK, 0)] = size;

        if D::RANK == 0 {
            inner_stride[0] = stride;
        } else {
            assert!(stride == self.inner_stride[0] * self.len() as isize, "invalid stride");

            shape[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(&self.shape[..]);
            inner_stride[0] = self.inner_stride[0];
        }

        FlatLayout::new(shape, inner_stride)
    }

    fn flatten(self) -> FlatLayout<U1, O> {
        let inner_stride = if D::RANK > 0 { self.inner_stride[0] } else { 1 };

        FlatLayout::new([self.len()], [inner_stride])
    }

    fn reformat<F: Format>(layout: Layout<D, F, O>) -> FlatLayout<D, O> {
        assert!(layout.is_uniformly_strided(), "array layout not uniformly strided");

        let mut inner_stride = <D::MaxOne as Dim>::Strides::default();

        if D::RANK > 0 {
            inner_stride[0] = layout.stride(D::dim::<O>(0));
        }

        FlatLayout::new(layout.shape(), inner_stride)
    }

    fn remove_dim(self, dim: usize) -> FlatLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == 0 || dim == D::RANK - 1, "invalid dimension");

        let mut shape = <D::Lower as Dim>::Shape::default();
        let mut inner_stride = <<D::Lower as Dim>::MaxOne as Dim>::Strides::default();

        if D::RANK > 1 {
            shape[..dim].copy_from_slice(&self.shape[..dim]);
            shape[dim..].copy_from_slice(&self.shape[dim + 1..]);

            let size = if dim == D::dim::<O>(0) { self.shape[dim] } else { 1 };

            inner_stride[0] = self.inner_stride[0] * size as isize;
        }

        FlatLayout::new(shape, inner_stride)
    }

    fn reshape<S: Shape>(self, shape: S) -> FlatLayout<S::Dim, O> {
        let len = shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(len == self.len(), "array length must not change");

        let mut inner_stride = <<S::Dim as Dim>::MaxOne as Dim>::Strides::default();

        if S::Dim::RANK > 0 {
            inner_stride[0] = if D::RANK > 0 { self.inner_stride[0] } else { 1 };
        }

        FlatLayout::new(shape, inner_stride)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> FlatLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == D::dim::<O>(D::RANK - 1), "invalid dimension");

        self.shape[dim] = size;

        FlatLayout::new(self.shape, self.inner_stride)
    }

    fn iter<T>(span: &SpanBase<T, FlatLayout<D, O>>) -> LinearIter<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIter::new_unchecked(span.as_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, FlatLayout<D, O>>) -> LinearIterMut<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIterMut::new_unchecked(span.as_mut_ptr(), layout.size(0), layout.stride(0)) }
    }
}

impl<D: Dim, O: Order> GeneralMapping<D, O> {
    pub fn new(shape: D::Shape, outer_strides: <D::Lower as Dim>::Strides) -> Self {
        Self { shape, outer_strides, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, General, O> for GeneralMapping<D, O> {
    fn has_linear_indexing(self) -> bool {
        D::RANK < 2
    }

    fn has_slice_indexing(self) -> bool {
        D::RANK < 2
    }

    fn is_contiguous(self) -> bool {
        if D::RANK > 1 {
            let mut stride = self.shape[D::dim::<O>(0)];

            for i in 1..D::RANK {
                if self.outer_strides[D::dim::<O>(i) - O::select(1, 0)] != stride as isize {
                    return false;
                }

                stride *= self.shape[D::dim::<O>(i)];
            }
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

        if D::RANK > 0 {
            strides[D::dim::<O>(0)] = 1;
            strides[D::dims::<O>(1..)].copy_from_slice(&self.outer_strides[..]);
        }

        strides
    }

    fn add_dim(self, size: usize, stride: isize) -> GeneralLayout<D::Higher, O> {
        StridedMapping::<D, O>::new(self.shape, self.strides()).add_dim(size, stride).reformat()
    }

    fn flatten(self) -> DenseLayout<U1, O> {
        assert!(self.is_contiguous(), "array layout not contiguous");

        DenseLayout::new([self.len()])
    }

    fn reformat<F: Format>(layout: Layout<D, F, O>) -> GeneralLayout<D, O> {
        assert!(D::RANK == 0 || layout.stride(D::dim::<O>(0)) == 1, "inner stride not unitary");

        let mut outer_strides = <D::Lower as Dim>::Strides::default();

        if D::RANK > 1 {
            outer_strides[..].copy_from_slice(&layout.strides()[D::dims::<O>(1..)]);
        }

        GeneralLayout::new(layout.shape(), outer_strides)
    }

    fn remove_dim(self, dim: usize) -> GeneralLayout<D::Lower, O> {
        assert!(D::RANK == 1 || dim != D::dim::<O>(0), "invalid dimension");

        StridedMapping::<D, O>::new(self.shape, self.strides()).remove_dim(dim).reformat()
    }

    fn reshape<S: Shape>(self, shape: S) -> GeneralLayout<S::Dim, O> {
        StridedMapping::<D, O>::new(self.shape, self.strides()).reshape(shape).reformat()
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> GeneralLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");

        self.shape[dim] = size;

        GeneralLayout::new(self.shape, self.outer_strides)
    }

    fn iter<T>(span: &SpanBase<T, GeneralLayout<D, O>>) -> Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, GeneralLayout<D, O>>) -> IterMut<'_, T> {
        span.as_mut_slice().iter_mut()
    }
}

impl<D: Dim, O: Order> StridedMapping<D, O> {
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { shape, strides, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Default for StridedMapping<D, O> {
    fn default() -> Self {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            strides[O::select(0, D::RANK - 1)] = 1;
        }

        Self { shape: D::Shape::default(), strides, phantom: PhantomData }
    }
}

impl<D: Dim, O: Order> Mapping<D, Strided, O> for StridedMapping<D, O> {
    fn has_linear_indexing(self) -> bool {
        D::RANK < 2
    }

    fn has_slice_indexing(self) -> bool {
        D::RANK == 0
    }

    fn is_contiguous(self) -> bool {
        self.strides[D::dim::<O>(0)] == 1 && self.is_uniformly_strided()
    }

    fn is_uniformly_strided(self) -> bool {
        if D::RANK > 1 {
            let mut stride = self.strides[D::dim::<O>(0)];

            for i in 1..D::RANK {
                stride *= self.shape[D::dim::<O>(i - 1)] as isize;

                if self.strides[D::dim::<O>(i)] != stride {
                    return false;
                }
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

    fn add_dim(self, size: usize, stride: isize) -> StridedLayout<D::Higher, O> {
        assert!(D::Higher::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Higher as Dim>::Shape::default();
        let mut strides = <D::Higher as Dim>::Strides::default();

        shape[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(&self.shape[..]);
        shape[O::select(D::RANK, 0)] = size;

        strides[O::select(0..D::RANK, 1..D::RANK + 1)].copy_from_slice(&self.strides[..]);
        strides[O::select(D::RANK, 0)] = stride;

        StridedLayout::new(shape, strides)
    }

    fn flatten(self) -> FlatLayout<U1, O> {
        assert!(self.is_uniformly_strided(), "array layout not uniformly strided");

        let inner_stride = if D::RANK > 0 { self.strides[D::dim::<O>(0)] } else { 1 };

        FlatLayout::new([self.len()], [inner_stride])
    }

    fn reformat<F: Format>(layout: Layout<D, F, O>) -> StridedLayout<D, O> {
        StridedLayout::new(layout.shape(), layout.strides())
    }

    fn remove_dim(self, dim: usize) -> StridedLayout<D::Lower, O> {
        assert!(D::RANK > 0, "invalid rank");

        let mut shape = <D::Lower as Dim>::Shape::default();
        let mut strides = <D::Lower as Dim>::Strides::default();

        if D::RANK > 1 {
            shape[..dim].copy_from_slice(&self.shape[..dim]);
            shape[dim..].copy_from_slice(&self.shape[dim + 1..]);

            strides[..dim].copy_from_slice(&self.strides[..dim]);
            strides[dim..].copy_from_slice(&self.strides[dim + 1..]);
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
            // Set strides for the next region or extend the current region.
            if old_len == new_len {
                old_stride = self.strides[D::dim::<O>(i)];
                new_stride = old_stride;
            } else {
                assert!(old_stride == self.strides[D::dim::<O>(i)], "memory layout not compatible");
            }

            old_len *= self.shape[D::dim::<O>(i)];
            old_stride *= self.shape[D::dim::<O>(i)] as isize;

            // Add dimensions within the current region.
            while k < S::Dim::RANK {
                let dim = O::select(k, S::Dim::RANK - 1 - k);
                let len = new_len.saturating_mul(shape[dim]);

                if len > old_len {
                    break;
                }

                strides[dim] = new_stride;

                new_len = len;
                new_stride *= shape[dim] as isize;

                k += 1;
            }
        }

        // Add remaining dimensions.
        while k < S::Dim::RANK {
            let dim = O::select(k, S::Dim::RANK - 1 - k);

            strides[dim] = new_stride;

            new_len = new_len.saturating_mul(shape[dim]);
            new_stride *= shape[dim] as isize;

            k += 1;
        }

        assert!(new_len == old_len, "array length must not change");

        StridedLayout::new(shape, strides)
    }

    fn resize_dim(mut self, dim: usize, size: usize) -> StridedLayout<D, O> {
        assert!(D::RANK > 0, "invalid rank");

        self.shape[dim] = size;

        StridedLayout::new(self.shape, self.strides)
    }

    fn iter<T>(span: &SpanBase<T, StridedLayout<D, O>>) -> LinearIter<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIter::new_unchecked(span.as_ptr(), layout.size(0), layout.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, StridedLayout<D, O>>) -> LinearIterMut<'_, T> {
        let layout = span.layout().flatten();

        unsafe { LinearIterMut::new_unchecked(span.as_mut_ptr(), layout.size(0), layout.stride(0)) }
    }
}
