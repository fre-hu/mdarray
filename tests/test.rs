use mdarray::Array;

#[test]
fn test_array() {
    let mut a = Array::new();

    a.resize([3, 4], 0);

    for i in 0..3 {
        for j in 0..4 {
            a[[i, j]] = i + j;
        }
    }

    a.iter_mut().for_each(|x| *x *= 2);

    assert_eq!(a[..].iter().sum::<usize>(), 60);
}
