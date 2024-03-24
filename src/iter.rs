use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;

use crate::dim::Dim;
use crate::expression::Expression;

/// Iterator type for array expressions.
#[derive(Clone)]
pub struct Iter<E: Expression> {
    expr: E,
    limit: <E::Dim as Dim>::Shape,
    index: <E::Dim as Dim>::Shape,
    done: bool,
}

impl<E: Expression> Iter<E> {
    pub(crate) fn new(expr: E) -> Self {
        let mut limit = <E::Dim as Dim>::Shape::default();
        let mut index = <E::Dim as Dim>::Shape::default();

        let done = expr.shape()[..].iter().product::<usize>() == 0;

        // Combine each dimension with its outer dimension if the split mask is not set.
        if E::Dim::RANK > 0 {
            let shape = expr.shape();
            let mut accum = shape[E::Dim::RANK - 1];

            for i in 1..E::Dim::RANK {
                if (E::SPLIT_MASK >> (E::Dim::RANK - 1 - i)) & 1 > 0 {
                    limit[E::Dim::RANK - i] = accum;
                    accum = 1;
                }

                accum *= shape[E::Dim::RANK - 1 - i];
            }

            limit[0] = accum;

            // Set innermost index to simplify checking in Iterator::next.
            if done {
                index[0] = accum;
            }
        }

        Self { expr, limit, index, done }
    }

    unsafe fn step_outer(&mut self) -> bool {
        for i in 1..E::Dim::RANK {
            if (E::SPLIT_MASK >> (i - 1)) & 1 > 0 {
                if self.index[i] + 1 < self.limit[i] {
                    self.expr.step_dim(i);
                    self.index[i] += 1;

                    return true;
                } else {
                    self.expr.reset_dim(i, self.index[i]);
                    self.index[i] = 0;
                }
            }
        }

        false
    }
}

impl<E: Expression + Debug> Debug for Iter<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let is_empty = self.expr.shape()[..].iter().product::<usize>() == 0;
        let is_default = self.index[..] == <E::Dim as Dim>::Shape::default()[..];

        assert!(is_empty || (is_default && !self.done), "iterator in use");

        f.debug_tuple("Iter").field(&self.expr).finish()
    }
}

impl<E: Expression> ExactSizeIterator for Iter<E> {}
impl<E: Expression> FusedIterator for Iter<E> {}

impl<E: Expression> Iterator for Iter<E> {
    type Item = E::Item;

    fn fold<T, F: FnMut(T, Self::Item) -> T>(mut self, init: T, mut f: F) -> T {
        if !self.done {
            if E::Dim::RANK > 0 {
                unsafe { fold::<T, E, <E::Dim as Dim>::Lower>(&mut self, init, &mut f) }
            } else {
                f(init, unsafe { self.expr.get_unchecked(0) })
            }
        } else {
            init
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        if E::Dim::RANK > 0 {
            if self.index[0] == self.limit[0] {
                if self.done || unsafe { !self.step_outer() } {
                    self.done = true;

                    return None;
                }

                self.index[0] = 0;
            }

            self.index[0] += 1;

            unsafe { Some(self.expr.get_unchecked(self.index[0] - 1)) }
        } else if !self.done {
            self.done = true;

            unsafe { Some(self.expr.get_unchecked(0)) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut len = 1;

        for i in 1..E::Dim::RANK {
            if (E::SPLIT_MASK >> (E::Dim::RANK - 1 - i)) & 1 > 0 {
                len = len * self.limit[E::Dim::RANK - i] - self.index[E::Dim::RANK - i];
            }
        }

        len = len * self.limit[0] - self.index[0];

        (len, Some(len))
    }
}

unsafe fn fold<T, E: Expression, I: Dim>(
    iter: &mut Iter<E>,
    mut accum: T,
    f: &mut impl FnMut(T, E::Item) -> T,
) -> T {
    if I::RANK > 0 {
        if E::SPLIT_MASK & (1 << (I::RANK - 1)) > 0 {
            loop {
                accum = fold::<T, E, I::Lower>(iter, accum, f);

                if iter.index[I::RANK] + 1 < iter.limit[I::RANK] {
                    iter.expr.step_dim(I::RANK);
                    iter.index[I::RANK] += 1;
                } else {
                    break;
                }
            }

            iter.expr.reset_dim(I::RANK, iter.index[I::RANK]);
            iter.index[I::RANK] = 0;
        } else {
            accum = fold::<T, E, I::Lower>(iter, accum, f);
        }
    } else {
        for i in iter.index[0]..iter.limit[0] {
            accum = f(accum, iter.expr.get_unchecked(i));
        }

        iter.index[0] = 0;
    }

    accum
}
