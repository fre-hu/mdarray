pub use crate::dimension::Dimension;
pub use crate::order::Order;
use std::marker::PhantomData;

pub trait Layout<const N: usize, const M: usize> {
    const ORDER: Order;

    fn inner_len(&self) -> usize;
    fn len(&self) -> usize;
    fn outer_size(&self, dim: usize) -> usize;
    fn outer_stride(&self, dim: usize) -> isize;
    fn shape(&self) -> &[usize; N];
    fn size(&self, dim: usize) -> usize;
    fn stride(&self, dim: usize) -> isize;
}

pub trait DenseLayout<const N: usize>: Layout<N, 0> {}

pub struct StaticLayout<D: Dimension<N>, const N: usize, const O: Order> {
    _dimension: PhantomData<D>,
}

pub struct StridedLayout<const N: usize, const M: usize, const O: Order> {
    shape: [usize; N],
    outer_strides: [isize; M],
}

impl<D: Dimension<N>, const N: usize, const O: Order> StaticLayout<D, N, O> {
    pub(crate) fn new() -> Self {
        Self {
            _dimension: PhantomData,
        }
    }
}

impl<const N: usize, const M: usize, const O: Order> StridedLayout<N, M, O> {
    pub(crate) fn new(shape: [usize; N], outer_strides: [isize; M]) -> Self {
        Self {
            shape,
            outer_strides,
        }
    }

    pub(crate) fn resize(&mut self, shape: [usize; N]) {
        self.shape = shape;
    }
}

impl<D: Dimension<N>, const N: usize, const O: Order> Layout<N, 0> for StaticLayout<D, N, O> {
    const ORDER: Order = O;

    fn inner_len(&self) -> usize {
        D::LEN
    }

    fn len(&self) -> usize {
        D::LEN
    }

    fn outer_size(&self, _: usize) -> usize {
        panic!()
    }

    fn outer_stride(&self, _: usize) -> isize {
        panic!()
    }

    fn shape(&self) -> &[usize; N] {
        &D::SHAPE
    }

    fn size(&self, dim: usize) -> usize {
        D::SHAPE[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        match O {
            Order::ColumnMajor => D::SHAPE[..dim].iter().product::<usize>() as isize,
            Order::RowMajor => D::SHAPE[dim + 1..].iter().product::<usize>() as isize,
        }
    }
}

impl<const N: usize, const M: usize, const O: Order> Layout<N, M> for StridedLayout<N, M, O> {
    const ORDER: Order = O;

    fn inner_len(&self) -> usize {
        match O {
            Order::ColumnMajor => self.shape[..N - M].iter().product(),
            Order::RowMajor => self.shape[M..].iter().product(),
        }
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn outer_size(&self, dim: usize) -> usize {
        match O {
            Order::ColumnMajor => self.shape[dim + (N - M)],
            Order::RowMajor => self.shape[M - 1 - dim],
        }
    }

    fn outer_stride(&self, dim: usize) -> isize {
        self.outer_strides[dim]
    }

    fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    fn stride(&self, dim: usize) -> isize {
        match O {
            Order::ColumnMajor => {
                if dim < N - M {
                    self.shape[..dim].iter().product::<usize>() as isize
                } else {
                    self.outer_strides[dim - (N - M)]
                }
            }
            Order::RowMajor => {
                if dim < M {
                    self.outer_strides[M - 1 - dim]
                } else {
                    self.shape[dim + 1..].iter().product::<usize>() as isize
                }
            }
        }
    }
}

impl<D: Dimension<N>, const N: usize, const O: Order> DenseLayout<N> for StaticLayout<D, N, O> {}
impl<const N: usize, const O: Order> DenseLayout<N> for StridedLayout<N, 0, O> {}
