use crate::order::Order;
use std::marker::PhantomData;

pub trait Layout<const N: usize, O: Order>: Copy {
    fn shape(&self) -> [usize; N];
    fn size(&self, dim: usize) -> usize;

    fn len(&self) -> usize {
        self.shape().iter().product()
    }
}

#[derive(Clone, Copy)]
pub struct StridedLayout<const N: usize, const M: usize, O: Order> {
    shape: [usize; N],
    strides: [isize; M],
    _marker: PhantomData<O>,
}

pub type DenseLayout<const N: usize, O> = StridedLayout<N, 0, O>;

impl<const N: usize, const M: usize, O: Order> StridedLayout<N, M, O> {
    pub const fn new(shape: [usize; N], strides: [isize; M]) -> Self {
        Self {
            shape,
            strides,
            _marker: PhantomData,
        }
    }

    pub fn stride(&self, dim: usize) -> isize {
        O::select(
            if dim < N - M {
                self.shape[..dim].iter().product::<usize>() as isize
            } else {
                self.strides[dim - (N - M)]
            },
            if dim < M {
                self.strides[dim]
            } else {
                self.shape[dim + 1..].iter().product::<usize>() as isize
            },
        )
    }

    pub fn strides(&self) -> [isize; M] {
        self.strides
    }
}

impl<const N: usize, const M: usize, O: Order> StridedLayout<N, M, O> {
    pub fn resize(&mut self, shape: [usize; N], strides: [isize; M]) {
        self.shape = shape;
        self.strides = strides;
    }
}

impl<const N: usize, const M: usize, O: Order> Layout<N, O> for StridedLayout<N, M, O> {
    fn shape(&self) -> [usize; N] {
        self.shape
    }

    fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }
}
