mod axis;
mod span;
mod view;

pub use axis::Axis;
pub use span::SpanIndex;
pub use view::{DimIndex, Params, ViewIndex};

#[cfg(not(feature = "nightly"))]
pub(crate) fn range<R>(range: R, bounds: std::ops::RangeTo<usize>) -> std::ops::Range<usize>
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

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}

#[cfg(not(feature = "nightly"))]
fn div_ceil(this: usize, rhs: usize) -> usize {
    let d = this / rhs;
    let r = this % rhs;

    if r > 0 && rhs > 0 {
        d + 1
    } else {
        d
    }
}
