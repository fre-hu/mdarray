#![allow(clippy::comparison_chain)]
#![allow(clippy::needless_range_loop)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(extern_types))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(impl_trait_in_assoc_type))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
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
use serde_test::{Token, assert_tokens};

#[cfg(feature = "nightly")]
use aligned_alloc::AlignedAlloc;
use mdarray::expr::{self, Apply, Expression, IntoExpression};
use mdarray::index::{Axis, Cols, Rows};
use mdarray::{Array, DTensor, DView, DViewMut, Tensor, View, ViewMut, array, tensor, view};
use mdarray::{Const, Dense, Dyn, DynRank, Layout, Rank, Shape, StepRange, Strided, step};
use mdarray::{DenseMapping, IntoCloned, Mapping, StridedMapping};

type U0 = Const<0>;
type U1 = Const<1>;
type U2 = Const<2>;
type U3 = Const<3>;

const U0: U0 = Const::<0>;
const U1: U1 = Const::<1>;
const U2: U2 = Const::<2>;
const U3: U3 = Const::<3>;

macro_rules! to_slice {
    ($slice:expr) => {
        $slice.to_tensor()[..]
    };
}

fn check_mapping<S: Shape, L: Layout, M: Mapping>(_: &M) {
    assert_eq!(any::type_name::<L::Mapping<S>>(), any::type_name::<M>());
}

fn check_view<L: Layout>() {
    let a = DTensor::<i32, 2>::from([[0]]);
    let a = a.remap::<(Dyn, Dyn), L>();

    check_mapping::<Rank<0>, L, _>(a.view(0, 0).mapping());
    check_mapping::<Rank<1>, Strided, _>(a.view(.., 0).mapping());
    check_mapping::<Rank<1>, Strided, _>(a.view(1.., 0).mapping());
    check_mapping::<Rank<1>, Strided, _>(a.view(sr(), 0).mapping());

    check_mapping::<Rank<1>, L, _>(a.view(0, ..).mapping());
    check_mapping::<Rank<2>, L, _>(a.view(.., ..).mapping());
    check_mapping::<Rank<2>, L, _>(a.view(1.., ..).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(sr(), ..).mapping());

    check_mapping::<Rank<1>, L, _>(a.view(0, 1..).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(.., 1..).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(1.., 1..).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(sr(), 1..).mapping());

    check_mapping::<Rank<1>, Strided, _>(a.view(0, sr()).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(.., sr()).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(1.., sr()).mapping());
    check_mapping::<Rank<2>, Strided, _>(a.view(sr(), sr()).mapping());
}

fn sr() -> StepRange<RangeFull, isize> {
    step(.., 2)
}

#[test]
fn test_base() {
    let mut a = DTensor::<usize, 3>::default();
    #[cfg(not(feature = "nightly"))]
    let mut b = DTensor::<usize, 3>::with_capacity(60);
    #[cfg(feature = "nightly")]
    let mut b = DTensor::<usize, 3>::with_capacity_in(60, a.allocator().clone());

    a.resize(&[3, 4, 5], 0);
    b.resize(&[5, 4, 3], 0);

    assert_eq!(a.dim(1), 4);
    assert_eq!(a.shape(), &Rank::<3>::from_dims(&[3, 4, 5]));
    assert_eq!(a.len(), 60);
    assert_eq!(a.rank(), 3);
    assert_eq!(a.stride(1), 5);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                b[[k, j, i]] = a[20 * i + 5 * j + k];
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

    assert_eq!(to_slice!(a.view(1.., ..2, 4)), [1104, 1114, 1204, 1214]);
    assert_eq!(to_slice!(b.view(4, ..2, 1..)), [1104, 1204, 1114, 1214]);

    assert_eq!(format!("{:?}", a.view(2, 1..3, ..2)), "[[1210, 1211], [1220, 1221]]");
    assert_eq!(format!("{:?}", b.view(..2, 1..3, 2)), "[[1210, 1220], [1211, 1221]]");

    assert_eq!(a.view(2, 1, ..), View::from([1210, 1211, 1212, 1213, 1214].as_slice()));
    assert_eq!(b.view(2, 1, ..), ViewMut::from([1012, 1112, 1212].as_mut_slice()));

    assert_eq!(a.view(1, 2..3, 3..), DView::<_, 2>::from(&[[1123, 1124]]));
    assert_eq!(b.view(3.., 2..3, 1), DViewMut::<_, 2>::from(&mut [[1123], [1124]]));

    assert_eq!(Tensor::from_elem([3, 4, 5], 1)[..], [1; 60]);

    assert_eq!(a, Tensor::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));
    assert_eq!(b, Tensor::from_fn([5, 4, 3], |i| 1000 + 100 * i[2] + 10 * i[1] + i[0]));

    assert_eq!(a.view(2, .., ..), a.axis_expr(0).into_iter().skip(2).next().unwrap());
    assert_eq!(b.tensor(2, .., ..), b.axis_expr_mut(U0).into_iter().skip(2).next().unwrap());

    assert_eq!(b.view(.., 2, ..), b.axis_expr(U1).into_iter().skip(2).next().unwrap());
    assert_eq!(a.tensor(.., 2, ..), a.axis_expr_mut(1).into_iter().skip(2).next().unwrap());

    assert_eq!(a.view(.., .., 2), a.axis_expr(2).into_iter().skip(2).next().unwrap());
    assert_eq!(b.tensor(.., .., 2), b.axis_expr_mut(U2).into_iter().skip(2).next().unwrap());

    assert_eq!(a.view(2, .., ..), a.outer_expr().into_iter().skip(2).next().unwrap());
    assert_eq!(b.tensor(2, .., ..), b.outer_expr_mut().into_iter().skip(2).next().unwrap());

    assert_eq!(a.contains(&1111), true);
    assert_eq!(a.view(1, 1.., 1..).contains(&9999), false);

    assert_eq!(a.view(1.., 2.., 3).into_diag(0), view![1123, 1233]);
    assert_eq!(a.view(2, 1.., ..).diag(0), view![1210, 1221, 1232]);
    assert_eq!(a.view_mut(1, 2.., 3..).diag_mut(0), view![1123, 1134]);

    assert_eq!(a.view(2, .., ..).into_col(1), view![1201, 1211, 1221, 1231]);
    assert_eq!(a.view(.., .., 1).col(2), view![1021, 1121, 1221]);
    assert_eq!(a.view_mut(2, .., ..).col_mut(1), view![1201, 1211, 1221, 1231]);

    assert_eq!(a.view(2, .., ..).into_row(1), view![1210, 1211, 1212, 1213, 1214]);
    assert_eq!(a.view(.., .., 1).row(2), view![1201, 1211, 1221, 1231]);
    assert_eq!(a.view_mut(2, .., ..).row_mut(1), view![1210, 1211, 1212, 1213, 1214]);

    assert_eq!(a.view(1, .., ..), a.at(1));
    assert_eq!(b.tensor(2, .., ..), b.expr_mut().into_dyn().at_mut(2));

    assert_eq!(a.view(2, .., ..), a.expr().into_dyn().into_at(2));
    assert_eq!(b.tensor(1, .., ..), b.expr_mut().into_at(1));

    assert_eq!(a.view(.., 2, ..), a.expr().into_dyn().axis_at(1, 2));
    assert_eq!(b.tensor(.., .., 1), b.axis_at_mut(U2, 1));

    assert_eq!(a.view(.., 2, ..), a.expr().into_axis_at(1, 2));
    assert_eq!(b.tensor(.., .., 1), b.expr_mut().into_dyn().into_axis_at(U2, 1));

    assert_eq!(Axis::index(1, 3), 1);
    assert_eq!(Axis::index(Const::<2>, 3), 2);
    assert_eq!(Axis::index(Cols, 3), 1);
    assert_eq!(Axis::index(Rows, 3), 2);

    assert_eq!(array![[1, 2, 3], [4, 5, 6]].array(1, ..), view![4, 5, 6]);
    assert_eq!(array![[1, 2, 3], [4, 5, 6]].view(.., 1).to_array(), view![2, 5]);

    assert_eq!((*array![[1, 2, 3], [4, 5, 6]]).to_owned(), array![[1, 2, 3], [4, 5, 6]]);
    assert_eq!((*view![[1, 2, 3], [4, 5, 6]]).to_owned(), view![[1, 2, 3], [4, 5, 6]]);

    let mut r = a.clone().into_shape([5, 4, 3]);
    let mut s = b.clone();

    unsafe {
        s.set_mapping(DenseMapping::new((3, 4, 5)));
    }

    a.resize(&[4, 4, 4], 9999);
    b.resize_with(&[4, 4, 4], || 9999);

    assert_eq!(a.flatten().iter().sum::<usize>(), 213576);
    assert_eq!(b.flatten().iter().sum::<usize>(), 213576);

    assert_eq!(r.view(1.., 1.., 1..).shape(), &(4, 3, 2));
    assert_eq!(s.view(1.., 1.., 1..).shape(), &(2, 3, 4));

    assert_eq!(r.view(1.., 1.., 1..).strides(), [12, 3, 1]);
    assert_eq!(s.view(1.., 1.., 1..).strides(), [20, 5, 1]);

    assert_eq!(r.view(1.., 1.., 1..).view(2, 1, 0)[[]], 1203);
    assert_eq!(s.view(1.., 1.., 1..).view(0, 1, 2)[[]], 1032);

    assert_eq!(Tensor::from_iter(0..10).tensor(step(.., 2))[..], [0, 2, 4, 6, 8]);
    assert_eq!(Tensor::from_iter(0..10).tensor(step(.., -2))[..], [9, 7, 5, 3, 1]);

    assert!(Tensor::from_iter(0..10).view(step(..0, isize::MAX)).is_empty());
    assert!(Tensor::from_iter(0..10).view_mut(step(..0, isize::MIN)).is_empty());

    assert_eq!(Tensor::from_iter(0..3).map(|x| 10 * x)[..], [0, 10, 20]);
    assert_eq!(Tensor::from_iter(0..3).apply(|x| 10 * x)[..], [0, 10, 20]);
    assert_eq!(Tensor::from_iter(0..3).zip_with(view![3, 4, 5], |(x, y)| x + y)[..], [3, 5, 7]);

    assert_eq!(to_slice!(a.view(.., ..2, ..2).split_at(1).0), [1000, 1001, 1010, 1011]);
    assert_eq!(to_slice!(a.view(..2, .., ..2).split_axis_at(U1, 3).1), [1030, 1031, 1130, 1131]);

    a.truncate(2);

    assert_eq!(to_slice!(a.view(.., ..2, ..2)), [1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111]);

    r.flatten_mut().iter_mut().for_each(|x| *x *= 2);
    s[..].iter_mut().for_each(|x| *x *= 2);

    assert_eq!(r.flatten().iter().sum::<usize>(), 134040);
    assert_eq!(s[..].iter().sum::<usize>(), 134040);

    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].into_flat(), view![1, 2, 3, 4, 5, 6]);
    assert_eq!(view![[1, 2, 3], [4, 5, 6]].into_flat(), view![1, 2, 3, 4, 5, 6]);

    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].into_dyn(), view![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].into_dyn().shape(), &DynRank::from_dims(&[2, 3]));

    assert_eq!(view![[1, 2, 3], [4, 5, 6]].into_dyn(), view![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(view![[1, 2, 3], [4, 5, 6]].into_dyn().shape(), &DynRank::from_dims(&[2, 3]));

    assert_eq!(tensor![[1, 2, 3]].into_mapping::<(U1, Dyn)>(), view![[1, 2, 3]]);
    assert_eq!(view![[1, 2, 3]].into_mapping::<(Dyn, U3), Dense>(), view![[1, 2, 3]]);

    assert_eq!(view![[1, 2, 3]].remap::<(U1, Dyn), Dense>(), view![[1, 2, 3]]);
    assert_eq!(tensor![[1, 2, 3]].remap_mut::<(Dyn, U3), Dense>(), view![[1, 2, 3]]);

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
    t.expand(&s.view(3.., .., ..));

    assert_eq!(Tensor::from_iter(s.into_shape([120])).as_ref(), t.into_vec());

    let mut d = DTensor::<_, 2>::from([[1, 2], [3, 4], [5, 6]]);
    let mut e = d.drain(1..2).eval();

    assert_eq!(d, Tensor::from(&array![[1, 2], [5, 6]]));
    assert_eq!(e, Tensor::<_, (U1, Dyn)>::from(&[[3, 4]]));

    assert_eq!(d, Tensor::from(array![[1, 2], [5, 6]]));
    assert_eq!(e, Tensor::<_, (Dyn, U2)>::from([[3, 4]]));

    let f1: [[_; 2]; 2] = TryFrom::try_from(tensor![[1, 2], [5, 6]]).unwrap();
    let f2: [[_; 2]; 1] = TryFrom::try_from(tensor![[3, 4]]).unwrap();

    assert!(f1 == [[1, 2], [5, 6]] && f2 == [[3, 4]]);

    assert_eq!(d, View::from(&array![[1, 2], [5, 6]]));
    assert_eq!(e, View::<_, (U1, Dyn)>::from(&[[3, 4]]));

    assert_eq!(d, ViewMut::from(&mut array![[1, 2], [5, 6]]));
    assert_eq!(e, ViewMut::<_, (Dyn, U2)>::from(&mut [[3, 4]]));

    let g1: &[[_; 2]; 2] = TryFrom::try_from(d.expr()).unwrap();
    let g2: &[[_; 2]; 1] = TryFrom::try_from(e.expr()).unwrap();

    assert!(*g1 == [[1, 2], [5, 6]] && *g2 == [[3, 4]]);

    let g3: &mut [[_; 2]; 2] = TryFrom::try_from(d.expr_mut()).unwrap();
    let g4: &mut [[_; 2]; 1] = TryFrom::try_from(e.expr_mut()).unwrap();

    assert!(*g3 == [[1, 2], [5, 6]] && *g4 == [[3, 4]]);

    assert_eq!(tensor![123].into_scalar(), Tensor::from_elem((), 123)[[]]);

    assert_eq!(DTensor::<i32, 3>::from([[[456; 0]; 3]; 0]).shape(), &(0, 3, 0));
    assert_eq!(DTensor::<i32, 3>::from([[[456; 3]; 0]; 3]).shape(), &(3, 0, 3));

    assert_eq!(view![1, 2, 3].permute(U0), view![1, 2, 3]);

    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].permute_mut((U0, 1)), view![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].permute_mut((1, U0)), view![[1, 4], [2, 5], [3, 6]]);

    let v = view![[[1, 2, 3], [4, 5, 6]]];

    assert_eq!(v.into_permuted((0, 1, 2)), view![[[1, 2, 3], [4, 5, 6]]]);
    assert_eq!(v.into_permuted((0, 2, 1)), view![[[1, 4], [2, 5], [3, 6]]]);
    assert_eq!(v.into_permuted(&[1, 0, 2]), view![[[1, 2, 3]], [[4, 5, 6]]]);
    assert_eq!(v.into_permuted(&[1, 2, 0]), view![[[1], [2], [3]], [[4], [5], [6]]]);
    assert_eq!(v.into_permuted(&[2, 0, 1][..]), view![[[1, 4]], [[2, 5]], [[3, 6]]]);
    assert_eq!(v.into_permuted(&[2, 1, 0][..]), view![[[1], [4]], [[2], [5]], [[3], [6]]]);

    assert_eq!(Array::<usize, ()>(123).into_scalar(), 123);
    assert_eq!(array![1, 2, 3].into_shape::<(U1, U3)>(), view![[1, 2, 3]]);
    assert_eq!(array![1, 2, 3].into_shape::<(U3, U1)>(), view![[1], [2], [3]]);
    assert_eq!(array![[1, 2, 3], [4, 5, 6]], view![[1, 2, 3], [4, 5, 6]]);

    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]].transpose(), view![[1, 4], [2, 5], [3, 6]]);
    assert_eq!(tensor![[1, 2, 3]].transpose_mut(), view![[1, 2, 3]].into_transposed());

    let mut h1 = array![[1, 2, 3], [4, 5, 6]];
    let h2: &mut [[_; 3]; 2] = h1.as_mut();
    let h3: &mut Array<_, (_, _)> = h2.as_mut();

    assert!(*h3 == array![[1, 2, 3], [4, 5, 6]]);

    let h4 = array![[1, 2, 3], [4, 5, 6]];
    let h5: &[[_; 3]; 2] = h4.as_ref();
    let h6: &Array<_, (_, _)> = h5.as_ref();

    assert!(*h6 == array![[1, 2, 3], [4, 5, 6]]);

    #[cfg(feature = "nightly")]
    let u = DTensor::<u8, 1, AlignedAlloc<64>>::with_capacity_in(64, AlignedAlloc::new(Global));

    #[cfg(feature = "nightly")]
    assert_eq!(u.as_ptr() as usize % 64, 0);
}

#[test]
fn test_expr() {
    let mut a = tensor![[1, 2, 3], [4, 5, 6]];

    assert_eq!((&a + &view![1, 2, 3]).eval()[..], [2, 4, 6, 5, 7, 9]);
    assert_eq!((&view![1, 2, 3] + &a).eval()[..], [2, 4, 6, 5, 7, 9]);

    assert_eq!(format!("{:?}", a.axis_expr(0)), "AxisExpr(0, [[1, 2, 3], [4, 5, 6]])");
    assert_eq!(format!("{:?}", a.outer_expr_mut()), "AxisExprMut(0, [[1, 2, 3], [4, 5, 6]])");

    assert_eq!(format!("{:?}", a.cols()), "Lanes(0, [[1, 2, 3], [4, 5, 6]])");
    assert_eq!(format!("{:?}", a.rows_mut()), "LanesMut(1, [[1, 2, 3], [4, 5, 6]])");

    assert_eq!(format!("{:?}", a.clone().drain(1..)), "IntoExpr([[4, 5, 6]])");
    assert_eq!(format!("{:?}", a.clone().into_expr()), "IntoExpr([[1, 2, 3], [4, 5, 6]])");

    assert_eq!(format!("{:?}", expr::fill(1)), "Fill(1)");
    assert_eq!(format!("{:?}", expr::fill_with(|| 1)), "FillWith");
    assert_eq!(format!("{:?}", expr::from_elem([1, 2], 3)), "FromElem((1, 2), 3)");
    assert_eq!(format!("{:?}", expr::from_fn([1, 2], |i| i[0])), "FromFn((1, 2))");

    let e1 = format!("{:?}", a.expr().cloned().map(|x| x + 3));
    let e2 = format!("{:?}", a.view(..1, ..).expr().zip(&a.view(1.., ..)));
    let e3 = format!("{:?}", a.view_mut(1.., ..).expr_mut().enumerate());

    assert_eq!(e1, "Map { expr: Cloned { expr: [[1, 2, 3], [4, 5, 6]] } }");
    assert_eq!(e2, "Zip { a: [[1, 2, 3]], b: [[4, 5, 6]] }");
    assert_eq!(e3, "Enumerate { expr: [[4, 5, 6]] }");

    assert_eq!(format!("{:?}", a.view(0, ..).iter()), "Iter([1, 2, 3])");
    assert_eq!(format!("{:?}", a.view_mut(1, ..).iter_mut()), "Iter([4, 5, 6])");

    assert_eq!(format!("{:?}", a.view(.., 1).iter()), "Iter([2, 5])");
    assert_eq!(format!("{:?}", a.view_mut(.., 2).iter_mut()), "Iter([3, 6])");

    let b = a.expr().copied().map(|x| x + 3).eval();

    assert_eq!(b, view![[4, 5, 6], [7, 8, 9]]);

    let mut c = tensor![[1, 2], [3, 4], [5, 6]];

    c.expand(tensor![[7, 8], [9, 10]].into_expr());
    _ = view![[11, 12]].expr().cloned().eval_into(&mut c);

    assert_eq!(c, view![[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]);

    c.assign(tensor![[1; 2]; 6].into_expr());

    assert_eq!(c, view![[1; 2]; 6]);

    c.assign(&view![[2; 2]; 6]);

    assert_eq!(c, view![[2; 2]; 6]);

    let d = view![[(1, 5), (2, 6)], [(3, 5), (4, 6)]];
    let e = tensor![[(0, 1), (1, 1)], [(2, 1), (3, 1)], [(4, 1), (5, 1)]];

    assert_eq!(expr::zip(&view![[1, 2], [3, 4]], &view![5, 6]).map(|(x, y)| (*x, *y)).eval(), d);
    assert_eq!(tensor![[1; 2]; 3].into_expr().enumerate().eval(), e);

    assert_eq!(a.cols().eval(), view![view![1, 4], view![2, 5], view![3, 6]]);
    assert_eq!(a.cols_mut().eval(), view![view![1, 4], view![2, 5], view![3, 6]]);

    assert_eq!(a.lanes(U0).eval(), view![view![1, 4], view![2, 5], view![3, 6]]);
    assert_eq!(a.lanes_mut(1).eval(), view![view![1, 2, 3], view![4, 5, 6]]);

    assert_eq!(a.rows().eval(), view![view![1, 2, 3], view![4, 5, 6]]);
    assert_eq!(a.rows_mut().eval(), view![view![1, 2, 3], view![4, 5, 6]]);
}

#[test]
fn test_hash() {
    let mut s1 = DefaultHasher::new();
    let mut s2 = DefaultHasher::new();

    DTensor::<usize, 3>::from([[[4, 5, 6], [7, 8, 9]]]).hash(&mut s1);

    for i in 0..9 {
        s2.write_usize(i + 1);
    }

    assert_eq!(s1.finish(), s2.finish());
}

#[test]
fn test_index() {
    check_view::<Dense>();
    check_view::<Strided>();
}

#[test]
fn test_macros() {
    let array1: Array<usize, _> = array![];
    let array2: Array<usize, _> = array![[]];
    let array3: Array<usize, _> = array![[[]]];
    let array4: Array<usize, _> = array![[[[]]]];
    let array5: Array<usize, _> = array![[[[[]]]]];
    let array6: Array<usize, _> = array![[[[[[]]]]]];

    let tensor1: Tensor<usize, _> = tensor![];
    let tensor2: Tensor<usize, _> = tensor![[]];
    let tensor3: Tensor<usize, _> = tensor![[[]]];
    let tensor4: Tensor<usize, _> = tensor![[[[]]]];
    let tensor5: Tensor<usize, _> = tensor![[[[[]]]]];
    let tensor6: Tensor<usize, _> = tensor![[[[[[]]]]]];

    let view1: View<usize, _> = view![];
    let view2: View<usize, _> = view![[]];
    let view3: View<usize, _> = view![[[]]];
    let view4: View<usize, _> = view![[[[]]]];
    let view5: View<usize, _> = view![[[[[]]]]];
    let view6: View<usize, _> = view![[[[[[]]]]]];

    assert_eq!(array1.shape(), &(U0,));
    assert_eq!(array2.shape(), &(U1, U0,));
    assert_eq!(array3.shape(), &(U1, U1, U0,));
    assert_eq!(array4.shape(), &(U1, U1, U1, U0,));
    assert_eq!(array5.shape(), &(U1, U1, U1, U1, U0,));
    assert_eq!(array6.shape(), &(U1, U1, U1, U1, U1, U0,));

    assert_eq!(tensor1.shape(), &(0,));
    assert_eq!(tensor2.shape(), &(1, 0));
    assert_eq!(tensor3.shape(), &(1, 1, 0));
    assert_eq!(tensor4.shape(), &(1, 1, 1, 0));
    assert_eq!(tensor5.shape(), &(1, 1, 1, 1, 0));
    assert_eq!(tensor6.shape(), &(1, 1, 1, 1, 1, 0));

    assert_eq!(view1.shape(), &(0,));
    assert_eq!(view2.shape(), &(1, 0));
    assert_eq!(view3.shape(), &(1, 1, 0));
    assert_eq!(view4.shape(), &(1, 1, 1, 0));
    assert_eq!(view5.shape(), &(1, 1, 1, 1, 0));
    assert_eq!(view6.shape(), &(1, 1, 1, 1, 1, 0));

    assert_eq!(tensor![1, 2, 3], array![1, 2, 3]);
    assert_eq!(tensor![[1, 2, 3], [4, 5, 6]], array![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(tensor![[[1, 2, 3], [4, 5, 6]]], array![[[1, 2, 3], [4, 5, 6]]]);
    assert_eq!(tensor![[[[1, 2, 3], [4, 5, 6]]]], array![[[[1, 2, 3], [4, 5, 6]]]]);
    assert_eq!(tensor![[[[[1, 2, 3], [4, 5, 6]]]]], array![[[[[1, 2, 3], [4, 5, 6]]]]]);
    assert_eq!(tensor![[[[[[1, 2, 3], [4, 5, 6]]]]]], array![[[[[[1, 2, 3], [4, 5, 6]]]]]]);

    assert_eq!(view![1, 2, 3], array![1, 2, 3]);
    assert_eq!(view![[1, 2, 3], [4, 5, 6]], array![[1, 2, 3], [4, 5, 6]]);
    assert_eq!(view![[[1, 2, 3], [4, 5, 6]]], array![[[1, 2, 3], [4, 5, 6]]]);
    assert_eq!(view![[[[1, 2, 3], [4, 5, 6]]]], array![[[[1, 2, 3], [4, 5, 6]]]]);
    assert_eq!(view![[[[[1, 2, 3], [4, 5, 6]]]]], array![[[[[1, 2, 3], [4, 5, 6]]]]]);
    assert_eq!(view![[[[[[1, 2, 3], [4, 5, 6]]]]]], array![[[[[[1, 2, 3], [4, 5, 6]]]]]]);

    assert_eq!(tensor![0; 1], array![0; 1]);
    assert_eq!(tensor![[0; 1]; 2], array![[0; 1]; 2]);
    assert_eq!(tensor![[[0; 1]; 2]; 3], array![[[0; 1]; 2]; 3]);
    assert_eq!(tensor![[[[0; 1]; 2]; 3]; 4], array![[[[0; 1]; 2]; 3]; 4]);
    assert_eq!(tensor![[[[[0; 1]; 2]; 3]; 4]; 5], array![[[[[0; 1]; 2]; 3]; 4]; 5]);
    assert_eq!(tensor![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], array![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6]);

    assert_eq!(view![0; 1], array![0; 1]);
    assert_eq!(view![[0; 1]; 2], array![[0; 1]; 2]);
    assert_eq!(view![[[0; 1]; 2]; 3], array![[[0; 1]; 2]; 3]);
    assert_eq!(view![[[[0; 1]; 2]; 3]; 4], array![[[[0; 1]; 2]; 3]; 4]);
    assert_eq!(view![[[[[0; 1]; 2]; 3]; 4]; 5], array![[[[[0; 1]; 2]; 3]; 4]; 5]);
    assert_eq!(view![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6], array![[[[[[0; 1]; 2]; 3]; 4]; 5]; 6]);
}

#[test]
fn test_mapping() {
    let d = DenseMapping::new((U1, 2, U3));
    let s = StridedMapping::new(DynRank::from_dims(&[1, 2, 3]), &[4, 5, 6]);

    assert_eq!(d.is_contiguous(), true);
    assert_eq!(s.is_empty(), false);
    assert_eq!(d.len(), 6);
    assert_eq!(s.rank(), 3);

    assert_eq!(d.dim(2), 3);
    assert_eq!(s.dims(), [1, 2, 3]);
    assert_eq!(d.stride(0), 6);
    assert_eq!(s.strides(), [4, 5, 6]);

    let x = format!("{:?}", d);
    let y = format!("{:?}", s);

    assert_eq!(x, "DenseMapping { shape: (Const(1), 2, Const(3)) }");
    assert_eq!(y, "StridedMapping { shape: DynRank([1, 2, 3]), strides: [4, 5, 6] }");
}

#[test]
fn test_ops() {
    let mut a = DTensor::<i32, 2>::from([[1, 2, 3], [4, 5, 6]]);
    let b = DTensor::<i32, 2>::from([[9, 8, 7], [6, 5, 4]]);

    a -= expr::fill(1);
    a -= &b;
    a -= b.expr();

    *a -= expr::fill(1);
    *a -= &b;
    *a -= b.expr();

    assert_eq!(a, tensor![[-37, -32, -27], [-22, -17, -12]]);

    a = a - expr::fill(1);
    a = a - &b;
    a = a - b.expr();

    a = expr::fill(1) - a;
    a = &b - a;
    a = b.expr() - a;

    assert_eq!(a, tensor![[57, 50, 43], [36, 29, 22]]);

    let mut a = a.into_shape([2, 3].as_slice());

    a = (&a - &b).eval();
    a = (&a - b.expr()).eval();
    a = (a.expr() - &b).eval();
    a = (a.expr() - b.expr()).eval();

    assert_eq!(a, tensor![[21, 18, 15], [12, 9, 6]]);

    a = (&a - expr::fill(1)).eval();
    a = (a.expr() - expr::fill(1)).eval();

    a = (expr::fill(1) - &a).eval();
    a = (expr::fill(1) - a.expr()).eval();

    assert_eq!(a, tensor![[19, 16, 13], [10, 7, 4]]);

    a = -a;
    a = (-&a).eval();
    a = (-a.expr()).eval();

    assert_eq!(a, tensor![[-19, -16, -13], [-10, -7, -4]]);

    assert!(a == a && *a == a && a == *a && *a == *a);
    assert!(a == a.expr() && a.expr() == a && a.expr() == a.expr());
    assert!(a == a.clone().expr_mut() && a.clone().expr_mut() == a);
    assert!(a.clone().expr_mut() == a.clone().expr_mut());

    let c = expr::fill_with(|| 1usize) + expr::from_elem([3, 2], 4);
    let c = c + expr::from_fn([3, 2], |x| x[0] + x[1]);

    assert_eq!(c.eval(), tensor![[5, 6], [6, 7], [7, 8]]);
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    assert_tokens(&Array::<i32, ()>(123), &[Token::I32(123)]);

    assert_tokens(&Array::<i32, (U0, U3)>([]), &[Token::Seq { len: Some(0) }, Token::SeqEnd]);

    assert_tokens(
        &Array::<i32, (U3, U0)>([[], [], []]),
        &[
            Token::Seq { len: Some(3) },
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::SeqEnd,
        ],
    );

    assert_tokens(
        &Array::<i32, (U1, U2, U3)>([[[4, 5, 6], [7, 8, 9]]]),
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

    assert_tokens(&Tensor::<_, _>::from_elem((), 123), &[Token::I32(123)]);

    assert_tokens(&Tensor::<i32, (Dyn, U3)>::new(), &[Token::Seq { len: Some(0) }, Token::SeqEnd]);

    assert_tokens(
        &Tensor::<i32, (U3, Dyn)>::new(),
        &[
            Token::Seq { len: Some(3) },
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::SeqEnd,
        ],
    );

    assert_tokens(
        &DTensor::<i32, 3>::from([[[4, 5, 6], [7, 8, 9]]]),
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
