#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(const_fn_trait_bound)]
#![feature(custom_inner_attributes)]
#![feature(generic_const_exprs)]
#![feature(ptr_metadata)]
#![feature(slice_ptr_len)]
#![feature(slice_range)]
#![warn(missing_docs)]

use mdarray::*;

fn to_vec<'a, T: 'a + Clone, I: Iterator<Item = &'a T>>(i: I) -> Vec<T> {
    i.cloned().collect::<Vec<T>>()
}

#[test]
fn test_mdarray() {
    let mut a = Array::default();
    let mut c = CArray::with_capacity_in(60, a.allocator().clone());

    a.resize([3, 4, 5], 0);
    c.resize([3, 4, 5], 0);

    assert_eq!(a.len(), 60);
    assert_eq!(a.shape(), &[3, 4, 5]);
    assert_eq!(a.size(1), 4);
    assert_eq!(a.stride(2), 12);
    assert_eq!(a.strides(), &[]);

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                a[[i, j, k]] = 1000 + 100 * i + 10 * j + k;
                c[[i, j, k]] = a[12 * k + 3 * j + i];
            }
        }
    }

    assert_eq!(to_vec(a.view(.., 2, 3).iter()), [1023, 1123, 1223]);
    assert_eq!(to_vec(a.view(1, 1.., 3).iter()), [1113, 1123, 1133]);
    assert_eq!(to_vec(a.view(1, 2, 2..).iter()), [1122, 1123, 1124]);

    assert_eq!(to_vec(a.view(1.., ..2, 4).iter()), [1104, 1204, 1114, 1214]);
    assert_eq!(to_vec(c.view(1.., ..2, 4).iter()), [1104, 1114, 1204, 1214]);

    let mut r = a.reshape([5, 4, 3]);
    let mut s = c.reshape([5, 4, 3]);

    assert_eq!(r.view(1.., 1.., 1..).shape(), &[4, 3, 2]);
    assert_eq!(s.view(1.., 1.., 1..).shape(), &[4, 3, 2]);

    assert_eq!(r.view(1.., 1.., 1..).strides(), &[5, 20]);
    assert_eq!(s.view(1.., 1.., 1..).strides(), &[12, 3]);

    assert_eq!(to_vec(r.view(1.., 1.., 1..).view(2, 1, 0).iter()), &[1032]);
    assert_eq!(to_vec(s.view(1.., 1.., 1..).view(2, 1, 0).iter()), &[1203]);

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

    assert_eq!(to_vec(s.iter()), to_vec(t.iter()));

    let u = SArray2::<usize, 3, 4>::new(5);
    let v = u.clone();

    assert_eq!(to_vec(u.iter()), to_vec(v.iter()));
}
