use std::fmt::{Debug, Formatter, Result};

use std::ops::{
    Bound, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::grid::Grid;
use crate::shape::Shape;
use crate::traits::FromExpression;

/// Array dimension trait.
pub trait Dim: Copy + Debug + Default + Send + Sync {
    /// Merge dimensions, where constant size is preferred over dynamic.
    type Merge<D: Dim>: Dim;

    /// The resulting type after conversion from an expression.
    type FromExpr<T, S: Shape>: FromExpression<T, S::Prepend<Self>>;

    /// Dimension size if known statically, or `None` if dynamic.
    const SIZE: Option<usize>;

    /// Creates an array dimension with the given size.
    ///
    /// # Panics
    ///
    /// Panics if the size is not matching a constant-sized dimension.
    fn from_size(size: usize) -> Self;

    /// Returns the number of elements in the dimension.
    fn size(self) -> usize;
}

/// Array dimensions trait.
pub trait Dims:
    Copy
    + Debug
    + Default
    + IndexMut<(Bound<usize>, Bound<usize>), Output = [usize]>
    + IndexMut<usize, Output = usize>
    + IndexMut<Range<usize>, Output = [usize]>
    + IndexMut<RangeFrom<usize>, Output = [usize]>
    + IndexMut<RangeFull, Output = [usize]>
    + IndexMut<RangeInclusive<usize>, Output = [usize]>
    + IndexMut<RangeTo<usize>, Output = [usize]>
    + IndexMut<RangeToInclusive<usize>, Output = [usize]>
    + Send
    + Sync
    + for<'a> TryFrom<&'a [usize], Error: Debug>
{
}

/// Array strides trait.
pub trait Strides:
    Copy
    + Debug
    + Default
    + IndexMut<(Bound<usize>, Bound<usize>), Output = [isize]>
    + IndexMut<usize, Output = isize>
    + IndexMut<Range<usize>, Output = [isize]>
    + IndexMut<RangeFrom<usize>, Output = [isize]>
    + IndexMut<RangeFull, Output = [isize]>
    + IndexMut<RangeInclusive<usize>, Output = [isize]>
    + IndexMut<RangeTo<usize>, Output = [isize]>
    + IndexMut<RangeToInclusive<usize>, Output = [isize]>
    + Send
    + Sync
    + for<'a> TryFrom<&'a [isize], Error: Debug>
{
}

/// Type-level constant.
#[derive(Clone, Copy, Default)]
pub struct Const<const N: usize>;

/// Dynamically-sized dimension type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dyn(pub usize);

impl<const N: usize> Debug for Const<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Const").field(&N).finish()
    }
}

impl<const N: usize> Dim for Const<N> {
    type Merge<D: Dim> = Self;
    type FromExpr<T, S: Shape> = <S::FromExpr<T> as FromExpression<T, S>>::WithConst<N>;

    const SIZE: Option<usize> = Some(N);

    fn from_size(size: usize) -> Self {
        assert!(size == N, "invalid size");

        Self
    }

    fn size(self) -> usize {
        N
    }
}

impl Dim for Dyn {
    type Merge<D: Dim> = D;
    type FromExpr<T, S: Shape> = Grid<T, S::Prepend<Self>>;

    const SIZE: Option<usize> = None;

    fn from_size(size: usize) -> Self {
        Self(size)
    }

    fn size(self) -> usize {
        self.0
    }
}

macro_rules! impl_dims_strides {
    ($($n:tt),+) => {
        $(
            impl Dims for [usize; $n] {}
            impl Strides for [isize; $n] {}
        )+
    };
}

impl_dims_strides!(0, 1, 2, 3, 4, 5, 6);
