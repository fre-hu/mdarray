#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(int_roundings))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![warn(missing_docs)]

#[cfg(feature = "nightly")]
mod aligned_alloc;

#[cfg(feature = "nightly")]
use std::alloc::Global;
use std::any;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::RangeFull;

#[cfg(feature = "serde")]
use serde_test::{assert_tokens, Token};

#[cfg(feature = "nightly")]
use aligned_alloc::AlignedAlloc;
use mdarray::{
    fill, grid, step, view, Const, Dense, DenseMapping, Dim, Flat, FlatMapping, General,
    GeneralMapping, Grid, Layout, Mapping, StepRange, Strided, StridedMapping, View, ViewMut,
};

macro_rules! to_slice {
    ($span:expr) => {
        $span.to_grid().as_slice()
    };
}

fn check_mapping<D: Dim, L: Layout, M: Mapping>(_: M) {
    assert_eq!(any::type_name::<L::Mapping<D>>(), any::type_name::<M>());
}

fn check_view<L: Layout>() {
    let a = Grid::from([[[[0]]]]);
    let a = a.remap::<L>();

    // a.view((_, _, 0, 0))

    check_mapping::<Const<0>, Dense, _>(a.view((0, 0, 0, 0)).mapping());
    check_mapping::<Const<1>, L::Uniform, _>(a.view((.., 0, 0, 0)).mapping());
    check_mapping::<Const<1>, L::Uniform, _>(a.view((1.., 0, 0, 0)).mapping());
    check_mapping::<Const<1>, Flat, _>(a.view((sr(), 0, 0, 0)).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view((0, .., 0, 0)).mapping());
    check_mapping::<Const<2>, L, _>(a.view((.., .., 0, 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., .., 0, 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), .., 0, 0)).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view((0, 1.., 0, 0)).mapping());
    check_mapping::<Const<2>, L, _>(a.view((.., 1.., 0, 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., 1.., 0, 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), 1.., 0, 0)).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view((0, sr(), 0, 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((.., sr(), 0, 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., sr(), 0, 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), sr(), 0, 0)).mapping());

    // a.view((_, _, .., 0))

    check_mapping::<Const<1>, Flat, _>(a.view((0, 0, .., 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((.., 0, .., 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., 0, .., 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), 0, .., 0)).mapping());

    check_mapping::<Const<2>, L::NonUnitStrided, _>(a.view((0, .., .., 0)).mapping());
    check_mapping::<Const<3>, L, _>(a.view((.., .., .., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., .., .., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), .., .., 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, 1.., .., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 1.., .., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 1.., .., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 1.., .., 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, sr(), .., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., sr(), .., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., sr(), .., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), sr(), .., 0)).mapping());

    // a.view((_, _, 1.., 0))

    check_mapping::<Const<1>, Flat, _>(a.view((0, 0, 1.., 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((.., 0, 1.., 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., 0, 1.., 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), 0, 1.., 0)).mapping());

    check_mapping::<Const<2>, L::NonUnitStrided, _>(a.view((0, .., 1.., 0)).mapping());
    check_mapping::<Const<3>, L, _>(a.view((.., .., 1.., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., .., 1.., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), .., 1.., 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, 1.., 1.., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 1.., 1.., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 1.., 1.., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 1.., 1.., 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, sr(), 1.., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., sr(), 1.., 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., sr(), 1.., 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), sr(), 1.., 0)).mapping());

    // a.view((_, _, sr(), 0))

    check_mapping::<Const<1>, Flat, _>(a.view((0, 0, sr(), 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((.., 0, sr(), 0)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., 0, sr(), 0)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), 0, sr(), 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, .., sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., .., sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., .., sr(), 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), .., sr(), 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, 1.., sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 1.., sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 1.., sr(), 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 1.., sr(), 0)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, sr(), sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., sr(), sr(), 0)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., sr(), sr(), 0)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), sr(), sr(), 0)).mapping());

    // a.view((_, _, 0, ..))

    check_mapping::<Const<1>, Flat, _>(a.view((0, 0, 0, ..)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((.., 0, 0, ..)).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view((1.., 0, 0, ..)).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view((sr(), 0, 0, ..)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, .., 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., .., 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., .., 0, ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), .., 0, ..)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, 1.., 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 1.., 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 1.., 0, ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 1.., 0, ..)).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view((0, sr(), 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., sr(), 0, ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., sr(), 0, ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), sr(), 0, ..)).mapping());

    // a.view((_, _, .., ..))

    check_mapping::<Const<2>, L::NonUnitStrided, _>(a.view((0, 0, .., ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 0, .., ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 0, .., ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 0, .., ..)).mapping());

    check_mapping::<Const<3>, L::NonUnitStrided, _>(a.view((0, .., .., ..)).mapping());
    check_mapping::<Const<4>, L, _>(a.view((.., .., .., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., .., .., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), .., .., ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, 1.., .., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., 1.., .., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., 1.., .., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), 1.., .., ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, sr(), .., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., sr(), .., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., sr(), .., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), sr(), .., ..)).mapping());

    // a.view((_, _, 1.., ..))

    check_mapping::<Const<2>, Strided, _>(a.view((0, 0, 1.., ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 0, 1.., ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 0, 1.., ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 0, 1.., ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, .., 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., .., 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., .., 1.., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), .., 1.., ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, 1.., 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., 1.., 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., 1.., 1.., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), 1.., 1.., ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, sr(), 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., sr(), 1.., ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., sr(), 1.., ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), sr(), 1.., ..)).mapping());

    // a.view((_, _, sr(), ..))

    check_mapping::<Const<2>, Strided, _>(a.view((0, 0, sr(), ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((.., 0, sr(), ..)).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view((1.., 0, sr(), ..)).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view((sr(), 0, sr(), ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, .., sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., .., sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., .., sr(), ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), .., sr(), ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, 1.., sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., 1.., sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., 1.., sr(), ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), 1.., sr(), ..)).mapping());

    check_mapping::<Const<3>, Strided, _>(a.view((0, sr(), sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((.., sr(), sr(), ..)).mapping());
    check_mapping::<Const<4>, L::NonUniform, _>(a.view((1.., sr(), sr(), ..)).mapping());
    check_mapping::<Const<4>, Strided, _>(a.view((sr(), sr(), sr(), ..)).mapping());
}

fn sr() -> StepRange<RangeFull, isize> {
    step(.., 2)
}

#[test]
fn test_base() {
    let mut a = Grid::default();
    #[cfg(not(feature = "nightly"))]
    let mut b = Grid::with_capacity(60);
    #[cfg(feature = "nightly")]
    let mut b = Grid::with_capacity_in(60, a.allocator().clone());

    a.resize([3, 4, 5], &0);
    b.resize([5, 4, 3], &0);

    assert_eq!(a.len(), 60);
    assert_eq!(a.shape(), [3, 4, 5]);
    assert_eq!(a.size(1), 4);
    assert_eq!(a.stride(2), 12);
    assert_eq!(a.strides(), [1, 3, 12]);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                b[[k, j, i]] = a.as_slice()[12 * k + 3 * j + i];
            }
        }
    }

    unsafe {
        assert_eq!(*a.get_unchecked([2, 3, 4]), 1234);

        *b.get_unchecked_mut([4, 3, 2]) = 1234;
    }

    assert_eq!(to_slice!(a.view((.., 2, 3))), [1023, 1123, 1223]);
    assert_eq!(to_slice!(a.view((1, 1.., 3))), [1113, 1123, 1133]);
    assert_eq!(to_slice!(a.view((1, 2, 2..))), [1122, 1123, 1124]);

    assert_eq!(to_slice!(a.view((1.., ..2, 4))), [1104, 1204, 1114, 1214]);
    assert_eq!(to_slice!(b.view((4, ..2, 1..))), [1104, 1114, 1204, 1214]);

    assert_eq!(format!("{:?}", a.view((2, 1..3, ..2))), "[[1210, 1220], [1211, 1221]]");
    assert_eq!(format!("{:?}", b.view((..2, 1..3, 2))), "[[1210, 1211], [1220, 1221]]");

    assert_eq!(a.view((2, 1, ..)), View::from([1210, 1211, 1212, 1213, 1214].as_slice()));
    assert_eq!(b.view((2, 1, ..)), ViewMut::from([1012, 1112, 1212].as_mut_slice()));

    assert_eq!(a.view((1, 2..3, 3..)), View::from(&[[1123], [1124]]));
    assert_eq!(b.view((3.., 2..3, 1)), ViewMut::from(&mut [[1123, 1124]]));

    assert_eq!(Grid::<usize, 3>::from_elem([3, 4, 5], &1).as_slice(), [1; 60]);

    assert_eq!(a, Grid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));
    assert_eq!(b, Grid::<usize, 3>::from_fn([5, 4, 3], |i| 1000 + 100 * i[2] + 10 * i[1] + i[0]));

    assert_eq!(a.view((2, .., ..)), a.axis_iter::<0>().skip(2).next().unwrap());
    assert_eq!(b.grid((2, .., ..)), b.axis_iter_mut::<0>().skip(2).next().unwrap());

    assert_eq!(b.view((.., 2, ..)), b.axis_iter::<1>().skip(2).next().unwrap());
    assert_eq!(a.grid((.., 2, ..)), a.axis_iter_mut::<1>().skip(2).next().unwrap());

    assert_eq!(a.view((.., .., 2)), a.axis_iter::<2>().skip(2).next().unwrap());
    assert_eq!(b.grid((.., .., 2)), b.axis_iter_mut::<2>().skip(2).next().unwrap());

    assert_eq!(b.view((2, .., ..)), b.inner_iter().skip(2).next().unwrap());
    assert_eq!(a.grid((2, .., ..)), a.inner_iter_mut().skip(2).next().unwrap());

    assert_eq!(a.view((.., .., 2)), a.outer_iter().skip(2).next().unwrap());
    assert_eq!(b.grid((.., .., 2)), b.outer_iter_mut().skip(2).next().unwrap());

    assert_eq!(a.contains(&1111), true);
    assert_eq!(a.view((1, 1.., 1..)).contains(&9999), false);

    let mut r = a.clone().into_shape([5, 4, 3]);
    let mut s = b.clone();

    unsafe {
        s.set_shape([3, 4, 5]);
    }

    a.resize([4, 4, 4], &9999);
    b.resize_with([4, 4, 4], || 9999);

    assert_eq!(a.flatten().iter().sum::<usize>(), 213576);
    assert_eq!(b.flatten().iter().sum::<usize>(), 213576);

    assert_eq!(r.view((1.., 1.., 1..)).shape(), [4, 3, 2]);
    assert_eq!(s.view((1.., 1.., 1..)).shape(), [2, 3, 4]);

    assert_eq!(r.view((1.., 1.., 1..)).strides(), [1, 5, 20]);
    assert_eq!(s.view((1.., 1.., 1..)).strides(), [1, 3, 12]);

    assert_eq!(r.view((1.., 1.., 1..)).view((2, 1, 0))[[]], 1032);
    assert_eq!(s.view((1.., 1.., 1..)).view((0, 1, 2))[[]], 1203);

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
    t.extend_from_span(&s.view((.., .., 5..)));

    assert_eq!(Grid::from_iter(s.into_shape([120])).as_ref(), t.into_vec());

    assert_eq!(grid![123].into_scalar(), grid![[123]].into_scalar());
    assert_eq!(View::from(&456)[[]], ViewMut::from(&mut 456)[0]);

    #[cfg(feature = "nightly")]
    let u = Grid::<u8, 1, AlignedAlloc<64>>::with_capacity_in(64, AlignedAlloc::new(Global));

    #[cfg(feature = "nightly")]
    assert_eq!(u.as_ptr() as usize % 64, 0);
}

#[test]
fn test_hash() {
    let mut s1 = DefaultHasher::new();
    let mut s2 = DefaultHasher::new();

    Grid::<usize, 3>::from([[[4], [5]], [[6], [7]], [[8], [9]]]).hash(&mut s1);

    for i in 0..9 {
        s2.write_usize(i + 1);
    }

    assert_eq!(s1.finish(), s2.finish());
}

#[test]
fn test_index() {
    check_view::<Dense>();
    check_view::<General>();
    check_view::<Flat>();
    check_view::<Strided>();
}

#[test]
fn test_iter() {
    let mut grid = Grid::<i32, 2>::from([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(format!("{:?}", grid.outer_iter()), "AxisIter([[1, 2, 3], [4, 5, 6]])");
    assert_eq!(format!("{:?}", grid.inner_iter_mut()), "AxisIterMut([[1, 4], [2, 5], [3, 6]])");

    assert_eq!(format!("{:?}", grid.view((.., 0)).iter()), "Iter([1, 2, 3])");
    assert_eq!(format!("{:?}", grid.view_mut((.., 1)).iter_mut()), "IterMut([4, 5, 6])");

    assert_eq!(format!("{:?}", grid.view((1, ..)).iter()), "FlatIter([2, 5])");
    assert_eq!(format!("{:?}", grid.view_mut((2, ..)).iter_mut()), "FlatIterMut([3, 6])");
}

#[test]
fn test_macros() {
    let grid1: Grid<usize, 1> = grid![];
    let grid2: Grid<usize, 2> = grid![[]];
    let grid3: Grid<usize, 3> = grid![[[]]];
    let grid4: Grid<usize, 4> = grid![[[[]]]];
    let grid5: Grid<usize, 5> = grid![[[[[]]]]];
    let grid6: Grid<usize, 6> = grid![[[[[[]]]]]];

    let view1: View<usize, 1> = view![];
    let view2: View<usize, 2> = view![[]];
    let view3: View<usize, 3> = view![[[]]];
    let view4: View<usize, 4> = view![[[[]]]];
    let view5: View<usize, 5> = view![[[[[]]]]];
    let view6: View<usize, 6> = view![[[[[[]]]]]];

    assert_eq!(grid1, Grid::new());
    assert_eq!(grid2, Grid::new());
    assert_eq!(grid3, Grid::new());
    assert_eq!(grid4, Grid::new());
    assert_eq!(grid5, Grid::new());
    assert_eq!(grid6, Grid::new());

    assert_eq!(view1, Grid::new());
    assert_eq!(view2, Grid::new());
    assert_eq!(view3, Grid::new());
    assert_eq!(view4, Grid::new());
    assert_eq!(view5, Grid::new());
    assert_eq!(view6, Grid::new());

    assert_eq!(grid![1, 2, 3], Grid::from([1, 2, 3]));
    assert_eq!(grid![[1, 2, 3], [4, 5, 6]], Grid::from([[1, 2, 3], [4, 5, 6]]));
    assert_eq!(grid![[[1, 2, 3], [4, 5, 6]]], Grid::from([[[1, 2, 3], [4, 5, 6]]]));
    assert_eq!(grid![[[[1, 2, 3], [4, 5, 6]]]], Grid::from([[[[1, 2, 3], [4, 5, 6]]]]));
    assert_eq!(grid![[[[[1, 2, 3], [4, 5, 6]]]]], Grid::from([[[[[1, 2, 3], [4, 5, 6]]]]]));
    assert_eq!(grid![[[[[[1, 2, 3], [4, 5, 6]]]]]], Grid::from([[[[[[1, 2, 3], [4, 5, 6]]]]]]));

    assert_eq!(view![1, 2, 3], View::from(&[1, 2, 3]));
    assert_eq!(view![[1, 2, 3], [4, 5, 6]], View::from(&[[1, 2, 3], [4, 5, 6]]));
    assert_eq!(view![[[1, 2, 3], [4, 5, 6]]], View::from(&[[[1, 2, 3], [4, 5, 6]]]));
    assert_eq!(view![[[[1, 2, 3], [4, 5, 6]]]], View::from(&[[[[1, 2, 3], [4, 5, 6]]]]));
    assert_eq!(view![[[[[1, 2, 3], [4, 5, 6]]]]], View::from(&[[[[[1, 2, 3], [4, 5, 6]]]]]));
    assert_eq!(view![[[[[[1, 2, 3], [4, 5, 6]]]]]], View::from(&[[[[[[1, 2, 3], [4, 5, 6]]]]]]));

    assert_eq!(grid![0; 1], Grid::from_elem([1], &0));
    assert_eq!(grid![[0; 1]; 2], Grid::from_elem([1, 2], &0));
    assert_eq!(grid![[[0; 1]; 2]; 3], Grid::from_elem([1, 2, 3], &0));
    assert_eq!(grid![[[[0; 1]; 2]; 3]; 4], Grid::from_elem([1, 2, 3, 4], &0));
    assert_eq!(grid![[[[[0; 1]; 2]; 3]; 4]; 5], Grid::from_elem([1, 2, 3, 4, 5], &0));
    assert_eq!(grid![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], Grid::from_elem([1, 2, 3, 4, 5, 6], &0));

    assert_eq!(view![0; 1], View::from(&[0; 1]));
    assert_eq!(view![[0; 1]; 2], View::from(&[[0; 1]; 2]));
    assert_eq!(view![[[0; 1]; 2]; 3], View::from(&[[[0; 1]; 2]; 3]));
    assert_eq!(view![[[[0; 1]; 2]; 3]; 4], View::from(&[[[[0; 1]; 2]; 3]; 4]));
    assert_eq!(view![[[[[0; 1]; 2]; 3]; 4]; 5], View::from(&[[[[[0; 1]; 2]; 3]; 4]; 5]));
    assert_eq!(view![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], View::from(&[[[[[[0; 1]; 2]; 3]; 4]; 5]; 6]));
}

#[test]
fn test_mapping() {
    let d = DenseMapping::<Const<3>>::new([1, 2, 3]);
    let f = FlatMapping::<Const<3>>::new([1, 2, 3], 4);
    let g = GeneralMapping::<Const<3>>::new([1, 2, 3], [4, 5]);
    let s = StridedMapping::<Const<3>>::new([1, 2, 3], [4, 5, 6]);

    assert_eq!(d.is_contiguous(), true);
    assert_eq!(f.is_empty(), false);
    assert_eq!(g.is_uniformly_strided(), false);
    assert_eq!(s.len(), 6);

    assert_eq!(d.shape(), [1, 2, 3]);
    assert_eq!(f.size(2), 3);
    assert_eq!(g.stride(0), 1);
    assert_eq!(s.strides(), [4, 5, 6]);

    assert_eq!(format!("{:?}", d), "DenseMapping { shape: [1, 2, 3] }");
    assert_eq!(format!("{:?}", f), "FlatMapping { shape: [1, 2, 3], inner_stride: 4 }");
    assert_eq!(format!("{:?}", g), "GeneralMapping { shape: [1, 2, 3], outer_strides: [4, 5] }");
    assert_eq!(format!("{:?}", s), "StridedMapping { shape: [1, 2, 3], strides: [4, 5, 6] }");
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
