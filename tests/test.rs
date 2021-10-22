#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(const_fn_trait_bound)]
#![feature(const_generics_defaults)]
#![feature(generic_const_exprs)]
#![feature(ptr_metadata)]
#![feature(slice_index_methods)]
#![feature(slice_ptr_len)]
#![feature(slice_range)]
#![feature(specialization)]
#![feature(trusted_len)]
#![warn(missing_docs)]

use mdarray::{CGrid, CView, Grid, SGrid1, SGrid2, View};
use std::cmp::Ordering;
use std::iter::FromIterator;

macro_rules! to_slice {
    ($view:expr) => {
        $view.to_grid().as_slice()
    };
}

#[test]
fn test_mdarray() {
    let mut a = Grid::default();
    let mut c = CGrid::with_capacity_in(60, a.allocator().clone());

    a.resize([3, 4, 5], 0);
    c.resize([3, 4, 5], 0);

    assert_eq!(a.len(), 60);
    assert_eq!(a.shape(), [3, 4, 5]);
    assert_eq!(a.size(1), 4);
    assert_eq!(a.stride(2), 12);
    assert_eq!(a.strides(), []);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                c[[i, j, k]] = a[12 * k + 3 * j + i];
            }
        }
    }

    assert_eq!(to_slice!(a.view(.., 2, 3)), [1023, 1123, 1223]);
    assert_eq!(to_slice!(a.view(1, 1.., 3)), [1113, 1123, 1133]);
    assert_eq!(to_slice!(a.view(1, 2, 2..)), [1122, 1123, 1124]);

    assert_eq!(to_slice!(a.view(1.., ..2, 4)), [1104, 1204, 1114, 1214]);
    assert_eq!(to_slice!(c.view(1.., ..2, 4)), [1104, 1114, 1204, 1214]);

    assert!(format!("{:?}", a.view(2, 1..3, ..2)) == "[[1210, 1220], [1211, 1221]]");
    assert!(format!("{:?}", c.view(2, 1..3, ..2)) == "[[1210, 1211], [1220, 1221]]");

    assert!(SGrid1::<usize, 3>::new(1).cmp(&SGrid1::<usize, 3>::new(2)) == Ordering::Less);
    assert!(a[..].cmp(&a[..]) == Ordering::Equal);

    assert!(a == a && *a == a && a == *a && *a == *a);
    assert!(a.view(1, .., 2) <= a.view(1, .., 2) && *a.view(1, .., 2) < a.view(2, .., 1));
    assert!(a.view(2, .., 1) > *a.view(1, .., 2) && *a.view(2, .., 1) >= *a.view(2, .., 1));

    assert!(&a.view(.., 1, 2) == AsRef::<View<usize, 1>>::as_ref(&[1012, 1112, 1212]));
    assert!(&a.view(1, 2..3, 3..) == AsRef::<View<usize, 2>>::as_ref(&[[1123], [1124]]));
    assert!(&c.view(1, 2..3, 3..) == AsRef::<CView<usize, 2>>::as_ref(&[[1123, 1124]]));

    let mut r = a.clone().reshape([5, 4, 3]);
    let mut s = c.clone().reshape([5, 4, 3]);

    a.resize([4, 4, 4], 9999);
    c.resize([4, 4, 4], 9999);

    assert_eq!(a.iter().sum::<usize>(), 213576);
    assert_eq!(c.iter().sum::<usize>(), 213576);

    assert_eq!(r.view(1.., 1.., 1..).shape(), [4, 3, 2]);
    assert_eq!(s.view(1.., 1.., 1..).shape(), [4, 3, 2]);

    assert_eq!(r.view(1.., 1.., 1..).strides(), [5, 20]);
    assert_eq!(s.view(1.., 1.., 1..).strides(), [12, 3]);

    assert_eq!(to_slice!(r.view(1.., 1.., 1..).view(2, 1, 0)), [1032]);
    assert_eq!(to_slice!(s.view(1.., 1.., 1..).view(2, 1, 0)), [1203]);

    r.iter_mut().for_each(|x| *x *= 2);
    s.as_mut_slice().iter_mut().for_each(|x| *x *= 2);

    assert_eq!(r.iter().sum::<usize>(), 134040);
    assert_eq!(s.as_slice().iter().sum::<usize>(), 134040);

    r.clear();

    assert!(r.is_empty());
    assert!(r.capacity() > 0);

    r.shrink_to_fit();

    assert!(r.capacity() == 0);

    let t = s.clone();

    assert_eq!(Grid::from_iter(s.drain()), Grid::from_iter(t.into_iter()));

    let u = SGrid2::<usize, 3, 4>::new(5);
    let v = u.clone();

    assert_eq!(u, v);
}
