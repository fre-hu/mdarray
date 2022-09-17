#![feature(allocator_api)]
#![feature(int_roundings)]
#![feature(slice_range)]
#![warn(missing_docs)]

#[cfg(feature = "nightly")]
mod aligned_alloc;

#[cfg(feature = "nightly")]
use std::alloc::Global;
use std::any;
use std::cmp::Ordering;
use std::ops::RangeFull;

#[cfg(feature = "serde")]
use serde_test::{assert_tokens, Token};

#[cfg(feature = "nightly")]
use aligned_alloc::AlignedAlloc;
use mdarray::{
    fill, step, CGrid, ColumnMajor, Dense, Dim, Flat, Format, General, Grid, Layout, Rank,
    StepRange, Strided, SubGrid, SubGridMut,
};

macro_rules! to_slice {
    ($span:expr) => {
        $span.to_grid().as_slice()
    };
}

fn check_layout<D: Dim, F: Format, L: Copy>(_: L) {
    assert_eq!(any::type_name::<Layout<D, F>>(), any::type_name::<L>());
}

fn check_view<F: Format>() {
    type Dim<const N: usize> = Rank<N, ColumnMajor>;

    let a = Grid::from([[[[0]]]]);
    let a = a.reformat::<F>();

    // a.view((_, _, 0, 0))

    check_layout::<Dim<0>, Dense, _>(a.view((0, 0, 0, 0)).layout());
    check_layout::<Dim<1>, F::Uniform, _>(a.view((.., 0, 0, 0)).layout());
    check_layout::<Dim<1>, F::Uniform, _>(a.view((1.., 0, 0, 0)).layout());
    check_layout::<Dim<1>, Flat, _>(a.view((sr(), 0, 0, 0)).layout());

    check_layout::<Dim<1>, Flat, _>(a.view((0, .., 0, 0)).layout());
    check_layout::<Dim<2>, F, _>(a.view((.., .., 0, 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., .., 0, 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), .., 0, 0)).layout());

    check_layout::<Dim<1>, Flat, _>(a.view((0, 1.., 0, 0)).layout());
    check_layout::<Dim<2>, F, _>(a.view((.., 1.., 0, 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., 1.., 0, 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), 1.., 0, 0)).layout());

    check_layout::<Dim<1>, Flat, _>(a.view((0, sr(), 0, 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((.., sr(), 0, 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., sr(), 0, 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), sr(), 0, 0)).layout());

    // a.view((_, _, .., 0))

    check_layout::<Dim<1>, Flat, _>(a.view((0, 0, .., 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((.., 0, .., 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., 0, .., 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), 0, .., 0)).layout());

    check_layout::<Dim<2>, F::NonUnitStrided, _>(a.view((0, .., .., 0)).layout());
    check_layout::<Dim<3>, F, _>(a.view((.., .., .., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., .., .., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), .., .., 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, 1.., .., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 1.., .., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 1.., .., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 1.., .., 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, sr(), .., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., sr(), .., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., sr(), .., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), sr(), .., 0)).layout());

    // a.view((_, _, 1.., 0))

    check_layout::<Dim<1>, Flat, _>(a.view((0, 0, 1.., 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((.., 0, 1.., 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., 0, 1.., 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), 0, 1.., 0)).layout());

    check_layout::<Dim<2>, F::NonUnitStrided, _>(a.view((0, .., 1.., 0)).layout());
    check_layout::<Dim<3>, F, _>(a.view((.., .., 1.., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., .., 1.., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), .., 1.., 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, 1.., 1.., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 1.., 1.., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 1.., 1.., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 1.., 1.., 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, sr(), 1.., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., sr(), 1.., 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., sr(), 1.., 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), sr(), 1.., 0)).layout());

    // a.view((_, _, sr(), 0))

    check_layout::<Dim<1>, Flat, _>(a.view((0, 0, sr(), 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((.., 0, sr(), 0)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., 0, sr(), 0)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), 0, sr(), 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, .., sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., .., sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., .., sr(), 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), .., sr(), 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, 1.., sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 1.., sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 1.., sr(), 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 1.., sr(), 0)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, sr(), sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., sr(), sr(), 0)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., sr(), sr(), 0)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), sr(), sr(), 0)).layout());

    // a.view((_, _, 0, ..))

    check_layout::<Dim<1>, Flat, _>(a.view((0, 0, 0, ..)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((.., 0, 0, ..)).layout());
    check_layout::<Dim<2>, F::NonUniform, _>(a.view((1.., 0, 0, ..)).layout());
    check_layout::<Dim<2>, Strided, _>(a.view((sr(), 0, 0, ..)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, .., 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., .., 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., .., 0, ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), .., 0, ..)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, 1.., 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 1.., 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 1.., 0, ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 1.., 0, ..)).layout());

    check_layout::<Dim<2>, Strided, _>(a.view((0, sr(), 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., sr(), 0, ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., sr(), 0, ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), sr(), 0, ..)).layout());

    // a.view((_, _, .., ..))

    check_layout::<Dim<2>, F::NonUnitStrided, _>(a.view((0, 0, .., ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 0, .., ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 0, .., ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 0, .., ..)).layout());

    check_layout::<Dim<3>, F::NonUnitStrided, _>(a.view((0, .., .., ..)).layout());
    check_layout::<Dim<4>, F, _>(a.view((.., .., .., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., .., .., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), .., .., ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, 1.., .., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., 1.., .., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., 1.., .., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), 1.., .., ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, sr(), .., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., sr(), .., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., sr(), .., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), sr(), .., ..)).layout());

    // a.view((_, _, 1.., ..))

    check_layout::<Dim<2>, Strided, _>(a.view((0, 0, 1.., ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 0, 1.., ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 0, 1.., ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 0, 1.., ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, .., 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., .., 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., .., 1.., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), .., 1.., ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, 1.., 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., 1.., 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., 1.., 1.., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), 1.., 1.., ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, sr(), 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., sr(), 1.., ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., sr(), 1.., ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), sr(), 1.., ..)).layout());

    // a.view((_, _, sr(), ..))

    check_layout::<Dim<2>, Strided, _>(a.view((0, 0, sr(), ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((.., 0, sr(), ..)).layout());
    check_layout::<Dim<3>, F::NonUniform, _>(a.view((1.., 0, sr(), ..)).layout());
    check_layout::<Dim<3>, Strided, _>(a.view((sr(), 0, sr(), ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, .., sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., .., sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., .., sr(), ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), .., sr(), ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, 1.., sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., 1.., sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., 1.., sr(), ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), 1.., sr(), ..)).layout());

    check_layout::<Dim<3>, Strided, _>(a.view((0, sr(), sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((.., sr(), sr(), ..)).layout());
    check_layout::<Dim<4>, F::NonUniform, _>(a.view((1.., sr(), sr(), ..)).layout());
    check_layout::<Dim<4>, Strided, _>(a.view((sr(), sr(), sr(), ..)).layout());
}

fn sr() -> StepRange<RangeFull, isize> {
    step(.., 2)
}

#[test]
fn test_base() {
    let mut a = Grid::default();
    #[cfg(not(feature = "nightly"))]
    let mut c = CGrid::with_capacity(60);
    #[cfg(feature = "nightly")]
    let mut c = CGrid::with_capacity_in(60, a.allocator().clone());

    a.resize([3, 4, 5], 0);
    c.resize([3, 4, 5], 0);

    assert_eq!(a.len(), 60);
    assert_eq!(a.shape(), [3, 4, 5]);
    assert_eq!(a.size(1), 4);
    assert_eq!(a.stride(2), 12);
    assert_eq!(a.strides(), [1, 3, 12]);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                c[[i, j, k]] = a.as_slice()[12 * k + 3 * j + i];
            }
        }
    }

    unsafe {
        assert_eq!(*a.get_unchecked([2, 3, 4]), 1234);

        *c.get_unchecked_mut([2, 3, 4]) = 1234;
    }

    assert_eq!(to_slice!(a.view((.., 2, 3))), [1023, 1123, 1223]);
    assert_eq!(to_slice!(a.view((1, 1.., 3))), [1113, 1123, 1133]);
    assert_eq!(to_slice!(a.view((1, 2, 2..))), [1122, 1123, 1124]);

    assert_eq!(to_slice!(a.view((1.., ..2, 4))), [1104, 1204, 1114, 1214]);
    assert_eq!(to_slice!(c.view((1.., ..2, 4))), [1104, 1114, 1204, 1214]);

    assert!(format!("{:?}", a.view((2, 1..3, ..2))) == "[[1210, 1220], [1211, 1221]]");
    assert!(format!("{:?}", c.view((2, 1..3, ..2))) == "[[1210, 1211], [1220, 1221]]");

    assert_eq!(a.view((2, 1, ..)), SubGrid::from([1210, 1211, 1212, 1213, 1214].as_slice()));
    assert_eq!(c.view((.., 1, 2)), SubGridMut::from([1012, 1112, 1212].as_mut_slice()));

    assert_eq!(a.view((1, 2..3, 3..)), SubGrid::from(&[[1123], [1124]]));
    assert_eq!(c.view((1, 2..3, 3..)), SubGridMut::from(&mut [[1123, 1124]]));

    assert_eq!(a, Grid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));
    assert_eq!(c, CGrid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));

    assert_eq!(a.view((2, .., ..)), a.axis_iter::<0>().skip(2).next().unwrap());
    assert_eq!(c.grid((2, .., ..)), c.axis_iter_mut::<0>().skip(2).next().unwrap());

    assert_eq!(c.view((.., 2, ..)), c.axis_iter::<1>().skip(2).next().unwrap());
    assert_eq!(a.grid((.., 2, ..)), a.axis_iter_mut::<1>().skip(2).next().unwrap());

    assert_eq!(a.view((.., .., 2)), a.axis_iter::<2>().skip(2).next().unwrap());
    assert_eq!(c.grid((.., .., 2)), c.axis_iter_mut::<2>().skip(2).next().unwrap());

    assert_eq!(c.view((.., .., 2)), c.inner_iter().skip(2).next().unwrap());
    assert_eq!(a.grid((2, .., ..)), a.inner_iter_mut().skip(2).next().unwrap());

    assert_eq!(a.view((.., .., 2)), a.outer_iter().skip(2).next().unwrap());
    assert_eq!(c.grid((2, .., ..)), c.outer_iter_mut().skip(2).next().unwrap());

    let mut r = a.clone().into_shape([5, 4, 3]);
    let mut s = c.clone();

    unsafe {
        s.set_layout(Layout::<_, Dense>::new([5, 4, 3]));
    }

    a.resize([4, 4, 4], 9999);
    c.resize_with([4, 4, 4], || 9999);

    assert_eq!(a.flatten().iter().sum::<usize>(), 213576);
    assert_eq!(c.flatten().iter().sum::<usize>(), 213576);

    assert_eq!(r.view((1.., 1.., 1..)).shape(), [4, 3, 2]);
    assert_eq!(s.view((1.., 1.., 1..)).shape(), [4, 3, 2]);

    assert_eq!(r.view((1.., 1.., 1..)).strides(), [1, 5, 20]);
    assert_eq!(s.view((1.., 1.., 1..)).strides(), [12, 3, 1]);

    assert_eq!(r.view((1.., 1.., 1..)).view((2, 1, 0))[[]], 1032);
    assert_eq!(s.view((1.., 1.., 1..)).view((2, 1, 0))[[]], 1203);

    assert_eq!(Grid::from_iter(0..10).grid(step(.., 2))[..], [0, 2, 4, 6, 8]);
    #[cfg(not(feature = "nightly"))]
    assert_eq!(Grid::from_iter(0..10).grid(step(.., -2))[..], [8, 6, 4, 2, 0]);
    #[cfg(feature = "nightly")]
    assert_eq!(Grid::from_iter(0..10).grid_in(step(.., -2), Global)[..], [8, 6, 4, 2, 0]);

    assert!(Grid::from_iter(0..10).view(step(..0, isize::MAX)).is_empty());
    assert!(Grid::from_iter(0..10).view_mut(step(..0, isize::MIN)).is_empty());

    assert_eq!(Grid::from_iter(0..3).map(|x| 10 * x)[..], [0, 10, 20]);

    assert_eq!(to_slice!(a.view((..2, ..2, ..)).split_at(1).0), [1000, 1100, 1010, 1110]);
    assert_eq!(to_slice!(a.view((..2, .., ..2)).split_axis_at::<1>(3).1), [1030, 1130, 1031, 1131]);

    a.truncate(2);

    assert_eq!(to_slice!(a.view((..2, ..2, ..))), [1000, 1100, 1010, 1110, 1001, 1101, 1011, 1111]);

    r.flatten_mut().iter_mut().for_each(|x| *x *= 2);
    s.as_mut_slice().iter_mut().for_each(|x| *x *= 2);

    assert_eq!(r.flatten().iter().sum::<usize>(), 134040);
    assert_eq!(s.as_slice().iter().sum::<usize>(), 134040);

    r.clear();

    assert!(r.is_empty());
    assert!(r.capacity() > 0);

    r.shrink_to_fit();

    assert!(r.capacity() == 0);

    let mut t = s.clone();

    s.reserve(60);
    t.reserve_exact(60);

    assert!(s.capacity() >= 120 && t.capacity() >= 120);

    s.shrink_to(60);
    t.shrink_to_fit();

    assert!(s.capacity() < 120 && t.capacity() < 120);

    t.try_reserve(usize::MAX).unwrap_err();
    t.try_reserve_exact(60).unwrap();

    s.append(&mut t.clone());
    t.extend_from_span(&s.view((5.., .., ..)));

    assert_eq!(Grid::from_iter(s.into_shape([120])).as_ref(), t.into_vec());

    #[cfg(feature = "nightly")]
    let u = Grid::<u8, 1, AlignedAlloc<64>>::with_capacity_in(64, AlignedAlloc::new(Global));

    #[cfg(feature = "nightly")]
    assert_eq!(u.as_ptr() as usize % 64, 0);
}

#[test]
fn test_index() {
    check_view::<Dense>();
    check_view::<General>();
    check_view::<Flat>();
    check_view::<Strided>();
}

#[test]
fn test_ops() {
    let mut a = Grid::<i32, 2>::from([[1, 2, 3], [4, 5, 6]]);
    let b = Grid::<i32, 2>::from([[9, 8, 7], [6, 5, 4]]);

    a -= fill(1);
    a -= &b;
    a -= b.as_span();

    *a.as_mut_span() -= fill(1);
    *a.as_mut_span() -= &b;
    *a.as_mut_span() -= b.as_span();

    assert_eq!(a, Grid::from([[-37, -32, -27], [-22, -17, -12]]));

    a = a - fill(1);
    a = a - &b;
    a = a - b.as_span();

    a = fill(1) - a;
    a = &b - a;
    a = b.as_span() - a;

    assert_eq!(a, Grid::from([[57, 50, 43], [36, 29, 22]]));

    a = &a - &b;
    a = &a - b.as_span();
    a = a.as_span() - &b;
    a = a.as_span() - b.as_span();

    assert_eq!(a, Grid::from([[21, 18, 15], [12, 9, 6]]));

    a = &a - fill(1);
    a = a.as_span() - fill(1);

    a = fill(1) - &a;
    a = fill(1) - a.as_span();

    assert_eq!(a, Grid::from([[19, 16, 13], [10, 7, 4]]));

    a = -a;
    a = -&a;
    a = -a.as_span();

    assert_eq!(a, Grid::from([[-19, -16, -13], [-10, -7, -4]]));

    assert!(Grid::from([1, 2, 3]).cmp(&Grid::from([4, 5])) == Ordering::Less);
    assert!(Grid::from([3]).as_span().cmp(Grid::from([2, 1]).as_span()) == Ordering::Greater);

    assert!(a == a && *a == a && a == *a && *a == *a);
    assert!(b.view((1, ..)) <= b.view((1, ..)) && *b.view((1, ..)) > b.view((2, ..)));
    assert!(b.view((2, ..)) < *b.view((1, ..)) && *b.view((2, ..)) >= *b.view((2, ..)));
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    assert_tokens(
        &Grid::<i32, 3>::from([[[4, 5, 6], [7, 8, 9]]]),
        &[
            Token::Seq { len: Some(1) },
            Token::Seq { len: Some(2) },
            Token::Seq { len: Some(3) },
            Token::I32(4),
            Token::I32(5),
            Token::I32(6),
            Token::SeqEnd,
            Token::Seq { len: Some(3) },
            Token::I32(7),
            Token::I32(8),
            Token::I32(9),
            Token::SeqEnd,
            Token::SeqEnd,
            Token::SeqEnd,
        ],
    );
}
