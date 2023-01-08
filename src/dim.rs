use std::fmt::Debug;

use std::ops::{
    Bound, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::format::{Dense, Format};

/// Array dimension trait, for rank and types for shape and strides.
pub trait Dim {
    /// Next higher dimension.
    type Higher: Dim;

    /// Next lower dimension.
    type Lower: Dim;

    /// Corresponding format based on the dimension.
    type Format<F: Format>: Format;

    /// Array shape type.
    type Shape: Shape<Dim = Self>;

    /// Array strides type.
    type Strides: Strides<Dim = Self>;

    /// Array rank, i.e. the number of dimensions.
    const RANK: usize;
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
pub struct Const<const N: usize>;

macro_rules! impl_dim {
    (($($n:tt),*), ($($format:ty),*)) => {
        $(
            impl Dim for Const<$n> {
                type Higher = Const<{ $n + ($n < 6) as usize }>;
                type Lower = Const<{ $n - ($n > 0) as usize }>;

                type Format<F: Format> = $format;

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
