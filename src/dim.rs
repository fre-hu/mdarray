use std::fmt::Debug;
use std::slice;

use std::ops::{
    Bound, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::order::Order;

/// Array dimension trait for rank, shape and strides.
pub trait Dim: Copy + Debug + Default {
    /// Next higher dimension.
    type Higher: Dim;

    /// Next lower dimension.
    type Lower: Dim;

    /// One if non-zero dimension and zero otherwise.
    type MaxOne: Dim;

    /// Array shape type.
    type Shape: Shape<Dim = Self>;

    /// Array strides type.
    type Strides: Strides<Dim = Self>;

    /// Array rank, i.e. the number of dimensions.
    const RANK: usize;

    /// Returns the dimension with the specified index, counted from the innermost dimension.
    fn dim<O: Order>(index: usize) -> usize {
        assert!(index < Self::RANK, "invalid dimension");

        O::select(index, Self::RANK - 1 - index)
    }

    /// Returns the dimensions with the specified indices, counted from the innermost dimension.
    fn dims<O: Order>(indices: impl RangeBounds<usize>) -> Range<usize> {
        let range = slice::range(indices, ..Self::RANK);

        O::select(range.clone(), Self::RANK - range.end..Self::RANK - range.start)
    }
}

/// Array shape trait.
pub trait Shape:
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
{
    /// Array dimension type.
    type Dim: Dim<Shape = Self>;
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
{
    /// Array dimension type.
    type Dim: Dim<Strides = Self>;
}

/// Type-level constant.
#[derive(Clone, Copy, Debug, Default)]
pub struct Const<const N: usize>;

pub(crate) type U0 = Const<0>;
pub(crate) type U1 = Const<1>;

macro_rules! impl_dimension {
    ($($n:tt),*) => {
        $(
            impl Dim for Const<$n> {
                type Higher = Const<{ $n + ($n < 6) as usize }>;
                type Lower = Const<{ $n - ($n > 0) as usize }>;
                type MaxOne = Const<{ ($n > 0) as usize }>;

                type Shape = [usize; $n];
                type Strides = [isize; $n];

                const RANK: usize = $n;
            }

            impl Shape for [usize; $n] {
                type Dim = Const<$n>;
            }

            impl Strides for [isize; $n] {
                type Dim = Const<$n>;
            }
        )*
    }
}

impl_dimension!(0, 1, 2, 3, 4, 5, 6);
