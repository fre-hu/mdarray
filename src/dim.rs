use std::fmt::Debug;
use std::marker::PhantomData;
use std::slice;

use std::ops::{
    Bound, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::format::{Dense, Format};
use crate::order::Order;

/// Array dimension trait, for rank and element order.
pub trait Dim: Copy + Debug + Default {
    /// Array element order.
    type Order: Order;

    /// Next higher dimension.
    type Higher: Dim<Order = Self::Order>;

    /// Next lower dimension.
    type Lower: Dim<Order = Self::Order>;

    /// Corresponding format based on the dimension.
    type Format<F: Format>: Format;

    /// Array shape type.
    type Shape: Shape<Dim<Self::Order> = Self>;

    /// Array strides type.
    type Strides: Strides<Dim<Self::Order> = Self>;

    /// Array rank, i.e. the number of dimensions.
    const RANK: usize;

    /// Returns the dimension with the specified index, counted from the innermost dimension.
    fn dim(index: usize) -> usize {
        assert!(index < Self::RANK, "invalid dimension");

        Self::Order::select(index, Self::RANK - 1 - index)
    }

    /// Returns the dimensions with the specified indices, counted from the innermost dimension.
    fn dims(indices: impl RangeBounds<usize>) -> Range<usize> {
        let range = slice::range(indices, ..Self::RANK);

        Self::Order::select(range.clone(), Self::RANK - range.end..Self::RANK - range.start)
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
    type Dim<O: Order>: Dim<Order = O, Shape = Self>;
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
    type Dim<O: Order>: Dim<Order = O, Strides = Self>;
}

/// Array rank type, including element order.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rank<const N: usize, O: Order> {
    phantom: PhantomData<O>,
}

macro_rules! impl_dim {
    (($($n:tt),*), ($($format:ty),*)) => {
        $(
            impl<O: Order> Dim for Rank<$n, O> {
                type Order = O;

                type Higher = Rank<{ $n + ($n < 6) as usize }, O>;
                type Lower = Rank<{ $n - ($n > 0) as usize }, O>;

                type Format<F: Format> = $format;

                type Shape = [usize; $n];
                type Strides = [isize; $n];

                const RANK: usize = $n;
            }

            impl Shape for [usize; $n] {
                type Dim<O: Order> = Rank<$n, O>;
            }

            impl Strides for [isize; $n] {
                type Dim<O: Order> = Rank<$n, O>;
            }
        )*
    }
}

impl_dim!((0, 1, 2, 3, 4, 5, 6), (Dense, F::Uniform, F, F, F, F, F));
