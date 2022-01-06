use std::iter::FusedIterator;
use std::slice::{Iter, IterMut};

use crate::dimension::{Dim, Shape};
use crate::format::Format;
use crate::iterator::{StridedIter, StridedIterMut};
use crate::layout::{DenseLayout, GeneralLayout, Layout, StridedLayout};
use crate::order::Order;
use crate::span::SpanBase;

pub trait Mapping<L: Layout> {
    type Larger: Layout<Dim = <L::Dim as Dim>::Larger, Format = L::Format, Order = L::Order>;
    type NonDense: Layout<Dim = L::Dim, Format = <L::Format as Format>::NonDense, Order = L::Order>;
    type Reshaped<S: Shape>: Layout<Dim = S::Dim, Format = L::Format, Order = L::Order>;
    type Smaller: Layout<Dim = <L::Dim as Dim>::Smaller, Format = L::Format, Order = L::Order>;
    type UnitStrided: Layout<
        Dim = L::Dim,
        Format = <L::Format as Format>::UnitStrided,
        Order = L::Order,
    >;

    type Iter<'a, T: 'a>: Clone
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a T>
    where
        L: 'a;

    type IterMut<'a, T: 'a>: DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a mut T>
    where
        L: 'a;

    fn add_dim(self, size: usize, stride: isize) -> Self::Larger;
    fn iter<T>(span: &SpanBase<T, L>) -> Self::Iter<'_, T>;
    fn iter_mut<T>(span: &mut SpanBase<T, L>) -> Self::IterMut<'_, T>;
    fn offset(&self, index: <L::Dim as Dim>::Shape) -> isize;
    fn remove_dim(self, dim: usize) -> Self::Smaller;
    fn reshape<S: Shape>(self, shape: S) -> Self::Reshaped<S>;
    fn to_dense(self) -> DenseLayout<L::Dim, L::Order>;
    fn to_non_dense(self) -> Self::NonDense;
    fn to_strided(self) -> StridedLayout<L::Dim, L::Order>;
    fn to_unit_strided(self) -> Self::UnitStrided;
}

impl<D: Dim, O: Order> Mapping<DenseLayout<D, O>> for DenseLayout<D, O> {
    type Larger = DenseLayout<D::Larger, O>;
    type NonDense = GeneralLayout<D, O>;
    type Reshaped<S: Shape> = DenseLayout<S::Dim, O>;
    type Smaller = DenseLayout<D::Smaller, O>;
    type UnitStrided = DenseLayout<D, O>;

    type Iter<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = Iter<'a, T>;

    type IterMut<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = IterMut<'a, T>;

    fn add_dim(self, size: usize, stride: isize) -> Self::Larger {
        assert!(D::Larger::RANK > D::RANK, "invalid rank");
        assert!(stride == self.len() as isize, "invalid stride");

        let mut shape = <D::Larger as Dim>::Shape::default();

        shape.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)]
            .copy_from_slice(self.shape().as_ref());
        shape.as_mut()[O::select(D::RANK, 0)] = size;

        Self::Larger::new(shape)
    }

    fn iter<T>(span: &SpanBase<T, DenseLayout<D, O>>) -> Self::Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, DenseLayout<D, O>>) -> Self::IterMut<'_, T> {
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

    fn remove_dim(self, dim: usize) -> Self::Smaller {
        assert!(D::RANK > 0, "invalid rank");
        assert!(dim == self.dim(D::RANK - 1), "invalid dimension");

        let mut shape = <D::Smaller as Dim>::Shape::default();

        if D::RANK > 1 {
            shape
                .as_mut()
                .copy_from_slice(&self.shape().as_ref()[O::select(0..D::RANK - 1, 1..D::RANK)]);
        }

        Self::Smaller::new(shape)
    }

    fn reshape<S: Shape>(self, shape: S) -> DenseLayout<S::Dim, O> {
        let len = shape.as_ref().iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        assert!(len == self.len(), "array length must not change");

        if len == 0 { DenseLayout::default() } else { DenseLayout::new(shape) }
    }

    fn to_dense(self) -> Self {
        self
    }

    fn to_non_dense(self) -> Self::NonDense {
        Self::NonDense::new(self.shape(), self.strides())
    }

    fn to_strided(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape(), self.strides())
    }

    fn to_unit_strided(self) -> Self {
        self
    }
}

impl<D: Dim, O: Order> Mapping<GeneralLayout<D, O>> for GeneralLayout<D, O> {
    type Larger = GeneralLayout<D::Larger, O>;
    type NonDense = GeneralLayout<D, O>;
    type Reshaped<S: Shape> = GeneralLayout<S::Dim, O>;
    type Smaller = GeneralLayout<D::Smaller, O>;
    type UnitStrided = GeneralLayout<D, O>;

    type Iter<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = Iter<'a, T>;

    type IterMut<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = IterMut<'a, T>;

    fn add_dim(self, size: usize, stride: isize) -> Self::Larger {
        self.to_strided().add_dim(size, stride).to_unit_strided()
    }

    fn iter<T>(span: &SpanBase<T, GeneralLayout<D, O>>) -> Self::Iter<'_, T> {
        span.as_slice().iter()
    }

    fn iter_mut<T>(span: &mut SpanBase<T, GeneralLayout<D, O>>) -> Self::IterMut<'_, T> {
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

    fn remove_dim(self, dim: usize) -> Self::Smaller {
        self.to_strided().remove_dim(dim).to_unit_strided()
    }

    fn reshape<S: Shape>(self, shape: S) -> GeneralLayout<S::Dim, O> {
        self.to_strided().reshape(shape).to_unit_strided()
    }

    fn to_dense(self) -> DenseLayout<D, O> {
        assert!(self.is_contiguous(), "array layout not contiguous");

        DenseLayout::new(self.shape())
    }

    fn to_non_dense(self) -> Self {
        self
    }

    fn to_strided(self) -> StridedLayout<D, O> {
        StridedLayout::new(self.shape(), self.strides())
    }

    fn to_unit_strided(self) -> Self {
        self
    }
}

impl<D: Dim, O: Order> Mapping<StridedLayout<D, O>> for StridedLayout<D, O> {
    type Larger = StridedLayout<D::Larger, O>;
    type NonDense = StridedLayout<D, O>;
    type Reshaped<S: Shape> = StridedLayout<S::Dim, O>;
    type Smaller = StridedLayout<D::Smaller, O>;
    type UnitStrided = GeneralLayout<D, O>;

    type Iter<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = StridedIter<'a, T>;

    type IterMut<'a, T: 'a>
    where
        D: 'a,
        O: 'a,
    = StridedIterMut<'a, T>;

    fn add_dim(self, size: usize, stride: isize) -> Self::Larger {
        assert!(D::Larger::RANK > D::RANK, "invalid rank");

        let mut shape = <D::Larger as Dim>::Shape::default();
        let mut strides = <D::Larger as Dim>::Strides::default();

        shape.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)]
            .copy_from_slice(self.shape().as_ref());
        shape.as_mut()[O::select(D::RANK, 0)] = size;

        strides.as_mut()[O::select(0..D::RANK, 1..D::RANK + 1)]
            .copy_from_slice(self.strides().as_ref());
        strides.as_mut()[O::select(D::RANK, 0)] = stride;

        Self::Larger::new(shape, strides)
    }

    fn iter<T>(span: &SpanBase<T, StridedLayout<D, O>>) -> Self::Iter<'_, T> {
        let flat = span.layout().reshape([span.len()]);

        unsafe { StridedIter::new(span.as_ptr(), flat.size(0), flat.stride(0)) }
    }

    fn iter_mut<T>(span: &mut SpanBase<T, StridedLayout<D, O>>) -> Self::IterMut<'_, T> {
        let flat = span.layout().reshape([span.len()]);

        unsafe { StridedIterMut::new(span.as_mut_ptr(), flat.size(0), flat.stride(0)) }
    }

    fn offset(&self, index: D::Shape) -> isize {
        let mut offset = 0;

        for i in 0..D::RANK {
            offset += self.stride(i) * index.as_ref()[i] as isize;
        }

        offset
    }

    fn remove_dim(self, dim: usize) -> Self::Smaller {
        assert!(D::RANK > 0, "invalid rank");

        let mut shape = <D::Smaller as Dim>::Shape::default();
        let mut strides = <D::Smaller as Dim>::Strides::default();

        if D::RANK > 1 {
            shape.as_mut()[..dim].copy_from_slice(&self.shape().as_ref()[..dim]);
            shape.as_mut()[dim..].copy_from_slice(&self.shape().as_ref()[dim + 1..]);

            strides.as_mut()[..dim].copy_from_slice(&self.strides().as_ref()[..dim]);
            strides.as_mut()[dim..].copy_from_slice(&self.strides().as_ref()[dim + 1..]);
        }

        Self::Smaller::new(shape, strides)
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

            if old_len == new_len {
                // Set strides for the next region.
                old_stride = self.stride(self.dim(i));
                new_stride = old_stride;
            } else {
                // Ensure consistent strides within a contiguous region.
                assert!(old_stride == self.stride(self.dim(i)), "invalid strides");
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

        if new_len == 0 {
            StridedLayout::new(S::default(), strides)
        } else {
            StridedLayout::new(shape, strides)
        }
    }

    fn to_dense(self) -> DenseLayout<D, O> {
        assert!(self.is_contiguous(), "array layout not contiguous");

        DenseLayout::new(self.shape())
    }

    fn to_non_dense(self) -> Self {
        self
    }

    fn to_strided(self) -> Self {
        self
    }

    fn to_unit_strided(self) -> GeneralLayout<D, O> {
        GeneralLayout::new(self.shape(), self.strides())
    }
}
