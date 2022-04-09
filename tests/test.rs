#![feature(allocator_api)]
#![feature(generic_associated_types)]
#![feature(int_roundings)]
#![feature(marker_trait_attr)]
#![feature(ptr_metadata)]
#![feature(slice_range)]
#![warn(missing_docs)]

mod aligned_alloc;

use std::alloc::Global;
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde_test::{assert_tokens, Token};

use aligned_alloc::AlignedAlloc;
use mdarray::{fill, step, CGrid, CSpan, Dense, Grid, Layout, Span};

macro_rules! to_slice {
    ($span:expr) => {
        $span.to_grid().as_slice()
    };
}

fn test_base() {
    let mut a = Grid::default();
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

    assert_eq!(to_slice!(a.view((.., 2, 3))), [1023, 1123, 1223]);
    assert_eq!(to_slice!(a.view((1, 1.., 3))), [1113, 1123, 1133]);
    assert_eq!(to_slice!(a.view((1, 2, 2..))), [1122, 1123, 1124]);

    assert_eq!(to_slice!(a.view((1.., ..2, 4))), [1104, 1204, 1114, 1214]);
    assert_eq!(to_slice!(c.view((1.., ..2, 4))), [1104, 1114, 1204, 1214]);

    assert!(format!("{:?}", a.view((2, 1..3, ..2))) == "[[1210, 1220], [1211, 1221]]");
    assert!(format!("{:?}", c.view((2, 1..3, ..2))) == "[[1210, 1211], [1220, 1221]]");

    assert!(&a.view((.., 1, 2)) == AsRef::<Span<usize, 1>>::as_ref(&[1012, 1112, 1212]));
    assert!(&a.view((1, 2..3, 3..)) == AsRef::<Span<usize, 2>>::as_ref(&[[1123], [1124]]));
    assert!(&c.view((1, 2..3, 3..)) == AsRef::<CSpan<usize, 2>>::as_ref(&[[1123, 1124]]));

    assert!(a == Grid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));
    assert!(c == CGrid::<usize, 3>::from_fn([3, 4, 5], |i| 1000 + 100 * i[0] + 10 * i[1] + i[2]));

    let mut r = a.clone().into_shape([5, 4, 3]);
    let mut s = c.clone();

    unsafe {
        s.set_layout(Layout::<_, Dense, _>::new([5, 4, 3]));
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
    assert_eq!(Grid::from_iter(0..10).grid_in(step(.., -2), Global)[..], [8, 6, 4, 2, 0]);

    assert!(Grid::from_iter(0..10).view(step(..0, isize::MAX)).is_empty());
    assert!(Grid::from_iter(0..10).view_mut(step(..0, isize::MIN)).is_empty());

    assert_eq!(Grid::from_iter(0..3).map(|x| 10 * x)[..], [0, 10, 20]);

    assert_eq!(to_slice!(a.view((..2, ..2, ..2)).split_outer(1).0), [1000, 1100, 1010, 1110]);
    assert_eq!(to_slice!(a.view_mut((..2, ..2, ..2)).split_axis(1, 1).1), [1010, 1110, 1011, 1111]);

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

    let u = Grid::<u8, 1, AlignedAlloc<64>>::with_capacity_in(64, AlignedAlloc::new(Global));

    assert_eq!(u.as_ptr() as usize % 64, 0);
}

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

#[test]
fn test_mdarray() {
    test_base();
    test_ops();

    #[cfg(feature = "serde")]
    test_serde();
}
