pub trait Dimension<const N: usize> {
    const LEN: usize;
    const SHAPE: [usize; N];
}

/// Shape for static 1-dimensional array.
pub struct Dim1<const X: usize>;

/// Shape for static 2-dimensional array.
pub struct Dim2<const X: usize, const Y: usize>;

impl<const X: usize> Dimension<1> for Dim1<X> {
    const LEN: usize = X;
    const SHAPE: [usize; 1] = [X];
}

impl<const X: usize, const Y: usize> Dimension<2> for Dim2<X, Y> {
    const LEN: usize = X * Y;
    const SHAPE: [usize; 2] = [X, Y];
}
