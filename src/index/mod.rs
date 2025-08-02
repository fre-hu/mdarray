//! Module for array slice and view indexing, and for array axis subarray types.

mod axis;
mod permutation;
mod slice;
mod view;

pub use axis::{Axis, Cols, Rows};
pub use permutation::Permutation;
pub use slice::SliceIndex;
pub use view::{DimIndex, ViewIndex};

#[doc(hidden)]
pub use axis::{Keep, Resize, Split};

#[cfg(not(feature = "nightly"))]
pub(crate) fn range<R>(range: R, bounds: core::ops::RangeTo<usize>) -> core::ops::Range<usize>
where
    R: core::ops::RangeBounds<usize>,
{
    let len = bounds.end;

    let start: core::ops::Bound<&usize> = range.start_bound();
    let start = match start {
        core::ops::Bound::Included(&start) => start,
        core::ops::Bound::Excluded(start) => start
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
        core::ops::Bound::Unbounded => 0,
    };

    let end: core::ops::Bound<&usize> = range.end_bound();
    let end = match end {
        core::ops::Bound::Included(end) => end
            .checked_add(1)
            .unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
        core::ops::Bound::Excluded(&end) => end,
        core::ops::Bound::Unbounded => len,
    };

    assert!(start <= end, "slice index starts at {start} but ends at {end}");
    assert!(end <= len, "range end index {end} out of range for slice of length {len}");

    core::ops::Range { start, end }
}

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}
