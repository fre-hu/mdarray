use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;

use crate::dim::Dims;
use crate::expression::Expression;
use crate::shape::Shape;

/// Iterator type for array expressions.
#[derive(Clone)]
pub struct Iter<E: Expression> {
    expr: E,
    inner_index: usize,
    inner_limit: usize,
    outer_index: <E::Shape as Shape>::Dims<usize>,
    outer_limit: <E::Shape as Shape>::Dims<usize>,
}

impl<E: Expression> Iter<E> {
    pub(crate) fn new(expr: E) -> Self {
        let outer_rank = expr.rank().saturating_sub(expr.inner_rank());

        let inner_index = 0;
        let inner_limit = expr.shape().with_dims(|dims| dims[outer_rank..].iter().product());

        let mut outer_index = Default::default();
        let mut outer_limit = Default::default();

        if outer_rank > 0 {
            outer_index = Dims::new(expr.rank());
            outer_limit =
                expr.shape().with_dims(|dims| TryFrom::try_from(dims).expect("invalid rank"));
        }

        Self { expr, inner_index, inner_limit, outer_index, outer_limit }
    }

    unsafe fn step_outer(&mut self) -> bool {
        let outer_rank = self.expr.rank().saturating_sub(self.expr.inner_rank());

        // If the inner rank is >0, reset the last dimension when stepping outer dimensions.
        // This is needed in the FromFn implementation.
        if outer_rank < self.expr.rank() {
            self.expr.reset_dim(self.expr.rank() - 1, 0);
        }

        for i in (0..outer_rank).rev() {
            if self.outer_index.as_ref()[i] + 1 < self.outer_limit.as_ref()[i] {
                self.expr.step_dim(i);
                self.outer_index.as_mut()[i] += 1;

                return true;
            } else {
                self.expr.reset_dim(i, self.outer_index.as_ref()[i]);
                self.outer_index.as_mut()[i] = 0;
            }
        }

        self.outer_index.as_mut().fill(0); // Ensure that following calls return false.

        false
    }
}

impl<E: Expression + Debug> Debug for Iter<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        assert!(self.inner_index == 0, "iterator in use");

        f.debug_tuple("Iter").field(&self.expr).finish()
    }
}

impl<E: Expression> ExactSizeIterator for Iter<E> {}
impl<E: Expression> FusedIterator for Iter<E> {}

impl<E: Expression> Iterator for Iter<E> {
    type Item = E::Item;

    fn fold<T, F: FnMut(T, Self::Item) -> T>(mut self, init: T, mut f: F) -> T {
        let mut accum = init;

        loop {
            for i in self.inner_index..self.inner_limit {
                accum = f(accum, unsafe { self.expr.get_unchecked(i) });
            }

            if unsafe { !self.step_outer() } {
                return accum;
            }

            self.inner_index = 0;
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner_index == self.inner_limit {
            if unsafe { !self.step_outer() } {
                return None;
            }

            self.inner_index = 0;
        }

        self.inner_index += 1;

        unsafe { Some(self.expr.get_unchecked(self.inner_index - 1)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let outer_rank = self.expr.rank().saturating_sub(self.expr.inner_rank());
        let mut len = 1;

        for i in 0..outer_rank {
            len = len * self.outer_limit.as_ref()[i] - self.outer_index.as_ref()[i];
        }

        len = len * self.inner_limit - self.inner_index;

        (len, Some(len))
    }
}
