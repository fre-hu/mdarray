#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(associated_type_defaults))]
#![cfg_attr(feature = "nightly", feature(extern_types))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(int_roundings))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![feature(impl_trait_in_assoc_type)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unused_results)]

#[cfg(feature = "nightly")]
pub mod aligned_alloc;

#[cfg(feature = "nightly")]
use std::alloc::Global;
use std::any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::RangeFull;

#[cfg(feature = "serde")]
use serde_test::{assert_tokens, Token};

#[cfg(feature = "nightly")]
use aligned_alloc::AlignedAlloc;
use mdarray::expr::{Expr, ExprMut};
use mdarray::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};
use mdarray::{expr, grid, step, Apply, DGrid, Expression, Grid, IntoExpression};
use mdarray::{Const, Dense, Dim, Flat, General, IntoCloned, Layout, StepRange, Strided};

macro_rules! to_slice {
    ($span:expr) => {
        $span.to_grid()[..]
    };
}

fn check_mapping<D: Dim, L: Layout, M: Mapping>(_: M) {
    assert_eq!(any::type_name::<L::Mapping<D>>(), any::type_name::<M>());
}

fn check_view<L: Layout>() {
    let a = DGrid::<i32, 3>::from([[[0]]]);
    let a = a.remap::<L>();

    // a.view(_, _, 0)

    check_mapping::<Const<0>, Dense, _>(a.view(0, 0, 0).mapping());
    check_mapping::<Const<1>, L::Uniform, _>(a.view(.., 0, 0).mapping());
    check_mapping::<Const<1>, L::Uniform, _>(a.view(1.., 0, 0).mapping());
    check_mapping::<Const<1>, Flat, _>(a.view(sr(), 0, 0).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view(0, .., 0).mapping());
    check_mapping::<Const<2>, L, _>(a.view(.., .., 0).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., .., 0).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), .., 0).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view(0, 1.., 0).mapping());
    check_mapping::<Const<2>, L, _>(a.view(.., 1.., 0).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., 1.., 0).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), 1.., 0).mapping());

    check_mapping::<Const<1>, Flat, _>(a.view(0, sr(), 0).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(.., sr(), 0).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., sr(), 0).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), sr(), 0).mapping());

    // a.view(_, _, ..)

    check_mapping::<Const<1>, Flat, _>(a.view(0, 0, ..).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(.., 0, ..).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., 0, ..).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), 0, ..).mapping());

    check_mapping::<Const<2>, L::NonUnitStrided, _>(a.view(0, .., ..).mapping());
    check_mapping::<Const<3>, L, _>(a.view(.., .., ..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., .., ..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), .., ..).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, 1.., ..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., 1.., ..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., 1.., ..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), 1.., ..).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, sr(), ..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., sr(), ..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., sr(), ..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), sr(), ..).mapping());

    // a.view(_, _, 1..)

    check_mapping::<Const<1>, Flat, _>(a.view(0, 0, 1..).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(.., 0, 1..).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., 0, 1..).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), 0, 1..).mapping());

    check_mapping::<Const<2>, L::NonUnitStrided, _>(a.view(0, .., 1..).mapping());
    check_mapping::<Const<3>, L, _>(a.view(.., .., 1..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., .., 1..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), .., 1..).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, 1.., 1..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., 1.., 1..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., 1.., 1..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), 1.., 1..).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, sr(), 1..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., sr(), 1..).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., sr(), 1..).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), sr(), 1..).mapping());

    // a.view(_, _, sr())

    check_mapping::<Const<1>, Flat, _>(a.view(0, 0, sr()).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(.., 0, sr()).mapping());
    check_mapping::<Const<2>, L::NonUniform, _>(a.view(1.., 0, sr()).mapping());
    check_mapping::<Const<2>, Strided, _>(a.view(sr(), 0, sr()).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, .., sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., .., sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., .., sr()).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), .., sr()).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, 1.., sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., 1.., sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., 1.., sr()).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), 1.., sr()).mapping());

    check_mapping::<Const<2>, Strided, _>(a.view(0, sr(), sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(.., sr(), sr()).mapping());
    check_mapping::<Const<3>, L::NonUniform, _>(a.view(1.., sr(), sr()).mapping());
    check_mapping::<Const<3>, Strided, _>(a.view(sr(), sr(), sr()).mapping());
}

fn sr() -> StepRange<RangeFull, isize> {
    step(.., 2)
}

#[test]
fn test_base() {
    let mut a = DGrid::<usize, 3>::default();
    #[cfg(not(feature = "nightly"))]
    let mut b = DGrid::<usize, 3>::with_capacity(60);
    #[cfg(feature = "nightly")]
    let mut b = DGrid::<usize, 3>::with_capacity_in(60, a.allocator().clone());

    a.resize([3, 4, 5], 0);
    b.resize([5, 4, 3], 0);

    assert_eq!(a.len(), 60);
    assert_eq!(a.rank(), 3);
    assert_eq!(a.shape(), [3, 4, 5]);
    assert_eq!(a.size(1), 4);
    assert_eq!(a.stride(2), 12);
    assert_eq!(a.strides(), [1, 3, 12]);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                b[[k, j, i]] = a[12 * k + 3 * j + i];
            }
        }
    }

    unsafe {
        assert_eq!(*a.get_unchecked([2, 3, 4]), 1234);

        *b.get_unchecked_mut([4, 3, 2]) = 1234;
    }

    assert_eq!(to_slice!(a.view(.., 2, 3)), [1023, 1123, 1223]);
    assert_eq!(to_slice!(a.view(1, 1.., 3)), [1113, 1123, 1133]);
    assert_eq!(to_slice!(a.view(1, 2, 2..)), [1122, 1123, 1124]);

    assert_eq!(to_slice!(a.view(1.., ..2, 4)), [1104, 1204, 1114, 1214]);
    assert_eq!(to_slice!(b.view(4, ..2, 1..)), [1104, 1114, 1204, 1214]);

    assert_eq!(format!("{:?}", a.view(2, 1..3, ..2)), "[[1210, 1220], [1211, 1221]]");
    assert_eq!(format!("{:?}", b.view(..2, 1..3, 2)), "[[1210, 1211], [1220, 1221]]");

    assert_eq!(a.view(2, 1, ..), Expr::from([1210, 1211, 1212, 1213, 1214].as_slice()));
    assert_eq!(b.view(2, 1, ..), ExprMut::from([1012, 1112, 1212].as_mut_slice()));

    assert_eq!(a.view(1, 2..3, 3..), Expr::from(&[[1123], [1124]]));
    assert_eq!(b.view(3.., 2..3, 1), ExprMut::from(&mut [[1123, 1124]]));

    assert_eq!(DGrid::<usize, 3>::from_elem([3, 4, 5], 1)[..], [1; 60]);

    assert_eq!(a, DGrid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));
    assert_eq!(b, DGrid::<usize, 3>::from_fn([5, 4, 3], |i| 1000 + 100 * i[2] + 10 * i[1] + i[0]));

    assert_eq!(a.view(2, .., ..), a.axis_expr::<0>().into_iter().skip(2).next().unwrap());
    assert_eq!(b.grid(2, .., ..), b.axis_expr_mut::<0>().into_iter().skip(2).next().unwrap());

    assert_eq!(b.view(.., 2, ..), b.axis_expr::<1>().into_iter().skip(2).next().unwrap());
    assert_eq!(a.grid(.., 2, ..), a.axis_expr_mut::<1>().into_iter().skip(2).next().unwrap());

    assert_eq!(a.view(.., .., 2), a.axis_expr::<2>().into_iter().skip(2).next().unwrap());
    assert_eq!(b.grid(.., .., 2), b.axis_expr_mut::<2>().into_iter().skip(2).next().unwrap());

    assert_eq!(a.view(.., .., 2), a.outer_expr().into_iter().skip(2).next().unwrap());
    assert_eq!(b.grid(.., .., 2), b.outer_expr_mut().into_iter().skip(2).next().unwrap());

    assert_eq!(a.contains(&1111), true);
    assert_eq!(a.view(1, 1.., 1..).contains(&9999), false);

    assert_eq!(a.view(1.., 2.., 3).into_diag(0), expr![1123, 1233]);
    assert_eq!(a.view(2, 1.., ..).diag(0), expr![1210, 1221, 1232]);
    assert_eq!(a.view_mut(1, 2.., 3..).diag_mut(0), expr![1123, 1134]);

    assert_eq!(a.view(.., .., 1).col(2), expr![1021, 1121, 1221]);
    assert_eq!(a.view_mut(2, .., ..).col_mut(1), expr![1201, 1211, 1221, 1231]);

    assert_eq!(a.view(.., .., 1).row(2), expr![1201, 1211, 1221, 1231]);
    assert_eq!(a.view_mut(2, .., ..).row_mut(1), expr![1210, 1211, 1212, 1213, 1214]);

    let mut r = a.clone().into_shape([5, 4, 3]);
    let mut s = b.clone();

    unsafe {
        s.set_mapping(DenseMapping::new([3, 4, 5]));
    }

    a.resize([4, 4, 4], 9999);
    b.resize_with([4, 4, 4], || 9999);

    assert_eq!(a.flatten().iter().sum::<usize>(), 213576);
    assert_eq!(b.flatten().iter().sum::<usize>(), 213576);

    assert_eq!(r.view(1.., 1.., 1..).shape(), [4, 3, 2]);
    assert_eq!(s.view(1.., 1.., 1..).shape(), [2, 3, 4]);

    assert_eq!(r.view(1.., 1.., 1..).strides(), [1, 5, 20]);
    assert_eq!(s.view(1.., 1.., 1..).strides(), [1, 3, 12]);

    assert_eq!(r.view(1.., 1.., 1..).view(2, 1, 0)[[]], 1032);
    assert_eq!(s.view(1.., 1.., 1..).view(0, 1, 2)[[]], 1203);

    assert_eq!(Grid::from_iter(0..10).grid(step(.., 2))[..], [0, 2, 4, 6, 8]);
    assert_eq!(Grid::from_iter(0..10).grid(step(.., -2))[..], [9, 7, 5, 3, 1]);

    assert!(Grid::from_iter(0..10).view(step(..0, isize::MAX)).is_empty());
    assert!(Grid::from_iter(0..10).view_mut(step(..0, isize::MIN)).is_empty());

    assert_eq!(Grid::from_iter(0..3).apply(|x| 10 * x)[..], [0, 10, 20]);
    assert_eq!(Grid::from_iter(0..3).zip_with(expr![3, 4, 5], |(x, y)| x + y)[..], [3, 5, 7]);

    assert_eq!(to_slice!(a.view(..2, ..2, ..).split_at(1).0), [1000, 1100, 1010, 1110]);
    assert_eq!(to_slice!(a.view(..2, .., ..2).split_axis_at::<1>(3).1), [1030, 1130, 1031, 1131]);

    a.truncate(2);

    assert_eq!(to_slice!(a.view(..2, ..2, ..)), [1000, 1100, 1010, 1110, 1001, 1101, 1011, 1111]);

    r.flatten_mut().iter_mut().for_each(|x| *x *= 2);
    s[..].iter_mut().for_each(|x| *x *= 2);

    assert_eq!(r.flatten().iter().sum::<usize>(), 134040);
    assert_eq!(s[..].iter().sum::<usize>(), 134040);

    r.clear();

    assert!(r.is_empty());
    assert!(r.capacity() > 0);

    r.shrink_to_fit();

    assert!(r.capacity() == 0);

    let mut t = s.clone();

    s.reserve(60);
    t.reserve_exact(60);

    assert!(s.capacity() >= 120 && t.capacity() >= 120);
    assert!(s.spare_capacity_mut().len() == s.capacity() - s.len());

    s.shrink_to(60);
    t.shrink_to_fit();

    assert!(s.capacity() < 120 && t.capacity() < 120);

    _ = t.try_reserve(usize::MAX).unwrap_err();
    t.try_reserve_exact(60).unwrap();

    s.append(&mut t.clone());
    t.expand(&s.view(.., .., 5..));

    assert_eq!(Grid::from_iter(s.into_shape([120])).as_ref(), t.into_vec());

    let mut d = DGrid::<usize, 2>::from([[1, 2], [3, 4], [5, 6]]);
    let e = Grid::from_expr(d.drain(1..2));

    assert_eq!(d, Expr::from(&[[1, 2], [5, 6]]));
    assert_eq!(e, Expr::from(&[[3, 4]]));

    assert_eq!(grid![123].into_scalar(), grid![[123]].into_scalar());

    assert_eq!(expr![1, 2, 3].permute::<0>(), expr![1, 2, 3]);

    assert_eq!(grid![[1, 2, 3], [4, 5, 6]].permute_mut::<0, 1>(), expr![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(grid![[1, 2, 3], [4, 5, 6]].permute_mut::<1, 0>(), expr![[1, 4], [2, 5], [3, 6]]);

    let v = expr![[[1, 2, 3], [4, 5, 6]]];

    assert_eq!(v.into_permuted::<0, 1, 2>(), expr![[[1, 2, 3], [4, 5, 6]]]);
    assert_eq!(v.into_permuted::<0, 2, 1>(), expr![[[1, 2, 3]], [[4, 5, 6]]]);
    assert_eq!(v.into_permuted::<1, 0, 2>(), expr![[[1, 4], [2, 5], [3, 6]]]);
    assert_eq!(v.into_permuted::<1, 2, 0>(), expr![[[1, 4]], [[2, 5]], [[3, 6]]]);
    assert_eq!(v.into_permuted::<2, 0, 1>(), expr![[[1], [2], [3]], [[4], [5], [6]]]);
    assert_eq!(v.into_permuted::<2, 1, 0>(), expr![[[1], [4]], [[2], [5]], [[3], [6]]]);

    #[cfg(feature = "nightly")]
    let u = DGrid::<u8, 1, AlignedAlloc<64>>::with_capacity_in(64, AlignedAlloc::new(Global));

    #[cfg(feature = "nightly")]
    assert_eq!(u.as_ptr() as usize % 64, 0);
}

#[test]
fn test_expr() {
    let mut a = grid![[1, 2, 3], [4, 5, 6]];

    assert_eq!(a.shape(), [3, 2]);

    assert_eq!((&a + &expr![1, 2, 3]).eval()[..], [2, 4, 6, 5, 7, 9]);
    assert_eq!((&expr![1, 2, 3] + &a).eval()[..], [2, 4, 6, 5, 7, 9]);

    assert_eq!(format!("{:?}", a.axis_expr::<0>()), "AxisExpr([[1, 4], [2, 5], [3, 6]])");
    assert_eq!(format!("{:?}", a.outer_expr_mut()), "AxisExprMut([[1, 2, 3], [4, 5, 6]])");

    assert_eq!(format!("{:?}", a.cols()), "Lanes([[1, 2, 3], [4, 5, 6]])");
    assert_eq!(format!("{:?}", a.rows_mut()), "LanesMut([[1, 4], [2, 5], [3, 6]])");

    assert_eq!(format!("{:?}", a.clone().drain(1..)), "Drain([[4, 5, 6]])");
    assert_eq!(format!("{:?}", a.clone().into_expr()), "IntoExpr([[1, 2, 3], [4, 5, 6]])");

    assert_eq!(format!("{:?}", expr::fill(1)), "Fill(1)");
    assert_eq!(format!("{:?}", expr::fill_with(|| 1)), "FillWith");
    assert_eq!(format!("{:?}", expr::from_elem([1, 2], 3)), "FromElem([1, 2], 3)");
    assert_eq!(format!("{:?}", expr::from_fn([1, 2], |i| i)), "FromFn([1, 2])");

    let e1 = format!("{:?}", a.expr().cloned().map(|x| x + 3));
    let e2 = format!("{:?}", a.view(.., ..1).expr().zip(&a.view(.., 1..)));
    let e3 = format!("{:?}", a.view_mut(.., 1..).expr_mut().enumerate());

    assert_eq!(e1, "Map { expr: Cloned { expr: [[1, 2, 3], [4, 5, 6]] } }");
    assert_eq!(e2, "Zip { a: [[1, 2, 3]], b: [[4, 5, 6]] }");
    assert_eq!(e3, "Enumerate { expr: [[4, 5, 6]] }");

    assert_eq!(format!("{:?}", a.view(.., 0).iter()), "Iter([1, 2, 3])");
    assert_eq!(format!("{:?}", a.view_mut(.., 1).iter_mut()), "Iter([4, 5, 6])");

    assert_eq!(format!("{:?}", a.view(1, ..).iter()), "Iter([2, 5])");
    assert_eq!(format!("{:?}", a.view_mut(2, ..).iter_mut()), "Iter([3, 6])");

    let b = a.expr().copied().map(|x| x + 3).eval();

    assert_eq!(b, expr![[4, 5, 6], [7, 8, 9]]);

    let mut c = grid![[1, 2], [3, 4], [5, 6]];

    c.expand(grid![[7, 8], [9, 10]].into_expr());
    _ = expr![11, 12].expr().cloned().eval_into(&mut c);

    assert_eq!(c, expr![[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]);

    c.assign(grid![[1; 2]; 6].into_expr());

    assert_eq!(c, expr![[1; 2]; 6]);

    c.assign(&expr![[2; 2]; 6]);

    assert_eq!(c, expr![[2; 2]; 6]);

    let d = expr![[(1, 5), (2, 6)], [(3, 5), (4, 6)]];
    let e = [[([0, 0], 1), ([1, 0], 1)], [([0, 1], 1), ([1, 1], 1)], [([0, 2], 1), ([1, 2], 1)]];

    assert_eq!(expr::zip(&expr![[1, 2], [3, 4]], &expr![5, 6]).map(|(x, y)| (*x, *y)).eval(), d);
    assert_eq!(grid![[1; 2]; 3].into_expr().enumerate().eval(), Grid::from(e));

    assert_eq!(a.cols().eval(), expr![expr![1, 2, 3], expr![4, 5, 6]]);
    assert_eq!(a.cols_mut().eval(), expr![expr![1, 2, 3], expr![4, 5, 6]]);

    assert_eq!(a.lanes::<0>().eval(), expr![expr![1, 2, 3], expr![4, 5, 6]]);
    assert_eq!(a.lanes_mut::<1>().eval(), expr![expr![1, 4], expr![2, 5], expr![3, 6]]);

    assert_eq!(a.rows().eval(), expr![expr![1, 4], expr![2, 5], expr![3, 6]]);
    assert_eq!(a.rows_mut().eval(), expr![expr![1, 4], expr![2, 5], expr![3, 6]]);
}

#[test]
fn test_hash() {
    let mut s1 = DefaultHasher::new();
    let mut s2 = DefaultHasher::new();

    DGrid::<usize, 3>::from([[[4], [5]], [[6], [7]], [[8], [9]]]).hash(&mut s1);

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
fn test_macros() {
    let grid1: Grid<usize, _> = grid![];
    let grid2: Grid<usize, _> = grid![[]];
    let grid3: Grid<usize, _> = grid![[[]]];
    let grid4: Grid<usize, _> = grid![[[[]]]];
    let grid5: Grid<usize, _> = grid![[[[[]]]]];
    let grid6: Grid<usize, _> = grid![[[[[[]]]]]];

    let view1: Expr<usize, _> = expr![];
    let view2: Expr<usize, _> = expr![[]];
    let view3: Expr<usize, _> = expr![[[]]];
    let view4: Expr<usize, _> = expr![[[[]]]];
    let view5: Expr<usize, _> = expr![[[[[]]]]];
    let view6: Expr<usize, _> = expr![[[[[[]]]]]];

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

    assert_eq!(expr![1, 2, 3], Expr::from(&[1, 2, 3]));
    assert_eq!(expr![[1, 2, 3], [4, 5, 6]], Expr::from(&[[1, 2, 3], [4, 5, 6]]));
    assert_eq!(expr![[[1, 2, 3], [4, 5, 6]]], Expr::from(&[[[1, 2, 3], [4, 5, 6]]]));
    assert_eq!(expr![[[[1, 2, 3], [4, 5, 6]]]], Expr::from(&[[[[1, 2, 3], [4, 5, 6]]]]));
    assert_eq!(expr![[[[[1, 2, 3], [4, 5, 6]]]]], Expr::from(&[[[[[1, 2, 3], [4, 5, 6]]]]]));
    assert_eq!(expr![[[[[[1, 2, 3], [4, 5, 6]]]]]], Expr::from(&[[[[[[1, 2, 3], [4, 5, 6]]]]]]));

    assert_eq!(grid![0; 1], Grid::from_elem([1], 0));
    assert_eq!(grid![[0; 1]; 2], Grid::from_elem([1, 2], 0));
    assert_eq!(grid![[[0; 1]; 2]; 3], Grid::from_elem([1, 2, 3], 0));
    assert_eq!(grid![[[[0; 1]; 2]; 3]; 4], Grid::from_elem([1, 2, 3, 4], 0));
    assert_eq!(grid![[[[[0; 1]; 2]; 3]; 4]; 5], Grid::from_elem([1, 2, 3, 4, 5], 0));
    assert_eq!(grid![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], Grid::from_elem([1, 2, 3, 4, 5, 6], 0));

    assert_eq!(expr![0; 1], Expr::from(&[0; 1]));
    assert_eq!(expr![[0; 1]; 2], Expr::from(&[[0; 1]; 2]));
    assert_eq!(expr![[[0; 1]; 2]; 3], Expr::from(&[[[0; 1]; 2]; 3]));
    assert_eq!(expr![[[[0; 1]; 2]; 3]; 4], Expr::from(&[[[[0; 1]; 2]; 3]; 4]));
    assert_eq!(expr![[[[[0; 1]; 2]; 3]; 4]; 5], Expr::from(&[[[[[0; 1]; 2]; 3]; 4]; 5]));
    assert_eq!(expr![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], Expr::from(&[[[[[[0; 1]; 2]; 3]; 4]; 5]; 6]));
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
    assert_eq!(d.rank(), 3);

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
    let mut a = DGrid::<i32, 2>::from([[1, 2, 3], [4, 5, 6]]);
    let b = DGrid::<i32, 2>::from([[9, 8, 7], [6, 5, 4]]);

    a -= expr::fill(1);
    a -= &b;
    a -= b.expr();

    *a -= expr::fill(1);
    *a -= &b;
    *a -= b.expr();

    assert_eq!(a, Grid::from([[-37, -32, -27], [-22, -17, -12]]));

    a = a - expr::fill(1);
    a = a - &b;
    a = a - b.expr();

    a = expr::fill(1) - a;
    a = &b - a;
    a = b.expr() - a;

    assert_eq!(a, Grid::from([[57, 50, 43], [36, 29, 22]]));

    a = (&a - &b).eval();
    a = (&a - b.expr()).eval();
    a = (a.expr() - &b).eval();
    a = (a.expr() - b.expr()).eval();

    assert_eq!(a, Grid::from([[21, 18, 15], [12, 9, 6]]));

    a = (&a - expr::fill(1)).eval();
    a = (a.expr() - expr::fill(1)).eval();

    a = (expr::fill(1) - &a).eval();
    a = (expr::fill(1) - a.expr()).eval();

    assert_eq!(a, Grid::from([[19, 16, 13], [10, 7, 4]]));

    a = -a;
    a = (-&a).eval();
    a = (-a.expr()).eval();

    assert_eq!(a, Grid::from([[-19, -16, -13], [-10, -7, -4]]));

    assert!(a == a && *a == a && a == *a && *a == *a);
    assert!(a == a.expr() && a.expr() == a && a.expr() == a.expr());
    assert!(a == a.clone().expr_mut() && a.clone().expr_mut() == a);
    assert!(a.clone().expr_mut() == a.clone().expr_mut());

    let c = expr::fill_with(|| 1usize) + expr::from_elem([2, 3], 4);
    let c = c + expr::from_fn([2, 3], |x| x[0] + x[1]);

    assert_eq!(c.eval(), Grid::from([[5, 6], [6, 7], [7, 8]]));
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    assert_tokens(
        &DGrid::<i32, 3>::from([[[4, 5, 6], [7, 8, 9]]]),
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

#[test]
fn test_traits() {
    let x = vec![1, 2, 3];
    let ptr = x.as_ptr();

    let y = x.into_cloned();
    let z: Vec<usize> = (&y).into_cloned();

    assert_eq!(ptr, y.as_ptr());
    assert_ne!(ptr, z.as_ptr());

    let mut u = vec![];
    let mut v = vec![];

    y.clone_to(&mut u);
    (&u).clone_to(&mut v);

    assert_eq!(ptr, u.as_ptr());
    assert_ne!(ptr, v.as_ptr());
}
