use std::fmt::Debug;
use std::marker::PhantomData;
#[cfg(feature = "nightly")]
use std::slice;

use std::ops::{
    Bound, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::format::{Dense, Format};
use crate::order::Order;

/// Array dimension trait, for rank and element order.
pub trait Dim {
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
    #[must_use]
    fn dim(index: usize) -> usize {
        assert!(index < Self::RANK, "invalid dimension");

        Self::Order::select(index, Self::RANK - 1 - index)
    }

    /// Returns the dimensions with the specified indices, counted from the innermost dimension.
    #[must_use]
    fn dims(indices: impl RangeBounds<usize>) -> Range<usize> {
        #[cfg(not(feature = "nightly"))]
        let range = range(indices, ..Self::RANK);
        #[cfg(feature = "nightly")]
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

#[cfg(not(feature = "nightly"))]
pub fn range<R>(range: R, bounds: std::ops::RangeTo<usize>) -> std::ops::Range<usize>
where
    R: std::ops::RangeBounds<usize>,
{
    let len = bounds.end;

    let start: std::ops::Bound<&usize> = range.start_bound();
    let start = match start {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(start) => start
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
        std::ops::Bound::Unbounded => 0,
    };

    let end: std::ops::Bound<&usize> = range.end_bound();
    let end = match end {
        std::ops::Bound::Included(end) => end
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => len,
    };

    assert!(start <= end, "slice index starts at {start} but ends at {end}");
    assert!(end <= len, "range end index {end} out of range for slice of length {len}");

    std::ops::Range { start, end }
}
