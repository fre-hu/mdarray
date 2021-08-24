pub trait Dimension<const N: usize> {
    const LEN: usize;
    const RANK: usize;
    const SHAPE: [usize; N];
}

/// Shape for static 1-dimensional array.
pub struct Dim1<const S0: usize>;

/// Shape for static 2-dimensional array.
pub struct Dim2<const S0: usize, const S1: usize>;

impl<const S0: usize> Dimension<1> for Dim1<S0> {
    const LEN: usize = S0;
    const RANK: usize = 1;
    const SHAPE: [usize; 1] = [S0];
}

impl<const S0: usize, const S1: usize> Dimension<2> for Dim2<S0, S1> {
    const LEN: usize = S0 * S1;
    const RANK: usize = 2;
    const SHAPE: [usize; 2] = [S0, S1];
}
