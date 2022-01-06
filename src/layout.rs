use std::fmt::Debug;
use std::marker::PhantomData;

use crate::dimension::{Const, Dim};
use crate::format::{Dense, Format, General, Strided};
use crate::mapping::Mapping;
use crate::order::Order;

/// Array layout, including rank, shape, strides and element order.
pub trait Layout: Copy + Debug + Default + Mapping<Self> {
    /// Array dimension type.
    type Dim: Dim;

    /// Array format type.
    type Format: Format;

    /// Array element order.
    type Order: Order;

    /// True if the array has dense layout.
    const IS_DENSE: bool;

    /// True if the array has dense or general layout.
    const IS_UNIT_STRIDED: bool;

    /// Returns true if the array elements are stored contiguously in memory.
    fn is_contiguous(&self) -> bool;

    /// Returns the shape of the array.
    fn shape(&self) -> <Self::Dim as Dim>::Shape;

    /// Returns the number of elements in the specified dimension.
    fn size(&self, dim: usize) -> usize;

    /// Returns the distance between elements in the specified dimension.
    fn stride(&self, dim: usize) -> isize;

    /// Returns the distance between elements in each dimension.
    fn strides(&self) -> <Self::Dim as Dim>::Strides;

    /// Returns the dimension with the specified index, counted from the innermost dimension.
    fn dim(&self, index: usize) -> usize {
        Self::Order::select(index, Self::Dim::RANK - 1 - index)
    }

    /// Returns true if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.shape().as_ref().iter().product()
    }

    /// Returns the rank of the array.
    fn rank(&self) -> usize {
        Self::Dim::RANK
    }
}

pub trait StaticLayout<D: Dim, O: Order> {
    const LAYOUT: DenseLayout<D, O>;
}

/// Dense array layout type.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseLayout<D: Dim, O: Order> {
    shape: D::Shape,
    _marker: PhantomData<O>,
}

/// General array layout type.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeneralLayout<D: Dim, O: Order> {
    shape: D::Shape,
    raw_strides: <D::Smaller as Dim>::Strides,
    _marker: PhantomData<O>,
}

/// Strided array layout type.
#[derive(Clone, Copy, Debug, Default)]
pub struct StridedLayout<D: Dim, O: Order> {
    shape: D::Shape,
    strides: D::Strides,
    _marker: PhantomData<O>,
}

impl<D: Dim, O: Order> DenseLayout<D, O> {
    /// Creates a new, dense array layout with the specified shape.
    pub fn new(shape: D::Shape) -> Self {
        Self { shape, _marker: PhantomData }
    }
}

impl<D: Dim, O: Order> Layout for DenseLayout<D, O> {
    type Dim = D;
    type Format = Dense;
    type Order = O;

    const IS_DENSE: bool = true;
    const IS_UNIT_STRIDED: bool = true;

    fn is_contiguous(&self) -> bool {
        true
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        self.shape.as_ref()[O::select(0..dim, dim + 1..D::RANK)].iter().product::<usize>() as isize
    }

    fn strides(&self) -> D::Strides {
        let mut strides = D::Strides::default();
        let mut stride = 1;

        for i in 0..D::RANK {
            strides.as_mut()[self.dim(i)] = stride as isize;
            stride *= self.shape.as_ref()[self.dim(i)];
        }

        strides
    }
}

impl<D: Dim, O: Order> GeneralLayout<D, O> {
    /// Creates a new, general array layout with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        assert!(
            D::RANK == 0 || strides.as_ref()[O::select(0, D::RANK - 1)] == 1,
            "inner stride not unitary"
        );

        let mut raw_strides = <D::Smaller as Dim>::Strides::default();

        if D::RANK > 1 {
            raw_strides
                .as_mut()
                .copy_from_slice(&strides.as_ref()[O::select(1..D::RANK, 0..D::RANK - 1)]);
        }

        Self { shape, raw_strides, _marker: PhantomData }
    }
}

impl<D: Dim, O: Order> Layout for GeneralLayout<D, O> {
    type Dim = D;
    type Format = General;
    type Order = O;

    const IS_DENSE: bool = false;
    const IS_UNIT_STRIDED: bool = true;

    fn is_contiguous(&self) -> bool {
        if D::RANK > 1 {
            let mut stride = self.shape.as_ref()[self.dim(0)];

            for i in 1..D::RANK {
                if self.raw_strides.as_ref()[self.dim(i) - O::select(1, 0)] != stride as isize {
                    return false;
                }

                stride *= self.shape.as_ref()[self.dim(i)]
            }
        }

        true
    }

    fn shape(&self) -> D::Shape {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape.as_ref()[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        if dim == self.dim(0) { 1 } else { self.raw_strides.as_ref()[dim - O::select(1, 0)] }
    }

    fn strides(&self) -> D::Strides {
        let mut strides = D::Strides::default();

        if D::RANK > 0 {
            strides.as_mut()[self.dim(0)] = 1;
            strides.as_mut()[O::select(1..D::RANK, 0..D::RANK - 1)]
                .copy_from_slice(self.raw_strides.as_ref());
        }

        strides
    }
}

impl<D: Dim, O: Order> StridedLayout<D, O> {
    /// Creates a new, strided array layout with the specified shape and strides.
    pub fn new(shape: D::Shape, strides: D::Strides) -> Self {
        Self { shape, strides, _marker: PhantomData }
    }
}

impl<D: Dim, O: Order> Layout for StridedLayout<D, O> {
    type Dim = D;
    type Format = Strided;
    type Order = O;

    const IS_DENSE: bool = false;
    const IS_UNIT_STRIDED: bool = false;

    fn is_contiguous(&self) -> bool {
        let mut stride = 1;

        for i in 0..D::RANK {
            if self.strides.as_ref()[self.dim(i)] != stride as isize {
                return false;
            }

            stride *= self.shape.as_ref()[self.dim(i)]
        }

        true
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
}

macro_rules! impl_static_layout {
    ($n:tt, ($($xyz:tt),+)) => {
        #[allow(unused_parens)]
        impl<O: Order, $(const $xyz: usize),+> StaticLayout<Const<$n>, O> for ($(Const<$xyz>),+) {
            const LAYOUT: DenseLayout<Const<$n>, O> =
                DenseLayout {
                    shape: [$($xyz),+],
                    _marker: PhantomData,
                };
        }
    };
}

impl_static_layout!(1, (X));
impl_static_layout!(2, (X, Y));
impl_static_layout!(3, (X, Y, Z));
impl_static_layout!(4, (X, Y, Z, W));
impl_static_layout!(5, (X, Y, Z, W, U));
impl_static_layout!(6, (X, Y, Z, W, U, V));
