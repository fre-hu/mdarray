use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;

use crate::expression::Expression;
use crate::shape::Shape;

/// Iterator type for array expressions.
#[derive(Clone)]
pub struct Iter<E: Expression> {
    expr: E,
    inner_index: usize,
    inner_limit: usize,
    outer_index: <<E::Shape as Shape>::Tail as Shape>::Dims,
    outer_limit: <<E::Shape as Shape>::Tail as Shape>::Dims,
}

impl<E: Expression> Iter<E> {
    pub(crate) fn new(expr: E) -> Self {
        let inner_index = 0;
        let mut inner_limit = 0;

        let outer_index = <<E::Shape as Shape>::Tail as Shape>::Dims::default();
        let mut outer_limit = <<E::Shape as Shape>::Tail as Shape>::Dims::default();

        if !expr.is_empty() {
            inner_limit = 1;

            if E::Shape::RANK > 0 {
                for i in (1..E::Shape::RANK).rev() {
                    let index = E::Shape::RANK - 1 - i;

                    inner_limit *= expr.dim(index);

                    if (E::SPLIT_MASK >> (i - 1)) & 1 > 0 {
                        outer_limit[index] = inner_limit;
                        inner_limit = 1;
                    }
                }

                inner_limit *= expr.dim(E::Shape::RANK - 1);
            }
        }

        Self { expr, inner_index, inner_limit, outer_index, outer_limit }
    }

    unsafe fn step_outer(&mut self) -> bool {
        for i in 1..E::Shape::RANK {
            let index = E::Shape::RANK - 1 - i;

            if (E::SPLIT_MASK >> (i - 1)) & 1 > 0 {
                if self.outer_index[index] + 1 < self.outer_limit[index] {
                    self.expr.step_dim(index);
                    self.outer_index[index] += 1;

                    return true;
                } else {
                    self.expr.reset_dim(index, self.outer_index[index]);
                    self.outer_index[index] = 0;
                }
            }
        }

        self.outer_limit = Default::default(); // Ensure that following calls return false.

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
        let mut len = 1;

        for i in (1..E::Shape::RANK).rev() {
            let index = E::Shape::RANK - 1 - i;

            if (E::SPLIT_MASK >> (i - 1)) & 1 > 0 {
                len = len * self.outer_limit[index] - self.outer_index[index];
            }
        }

        len = len * self.inner_limit - self.inner_index;

        (len, Some(len))
    }
}
