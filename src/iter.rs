use std::fmt::{Debug, Formatter, Result};
use std::iter::FusedIterator;

use crate::dim::Dim;
use crate::expr::Producer;

/// Multidimensional array iterator type.
#[derive(Clone)]
pub struct Iter<P: Producer> {
    producer: P,
    limit: <P::Dim as Dim>::Shape,
    index: <P::Dim as Dim>::Shape,
    done: bool,
}

impl<P: Producer> Iter<P> {
    pub(crate) fn new(producer: P) -> Self {
        let mut limit = <P::Dim as Dim>::Shape::default();
        let mut index = <P::Dim as Dim>::Shape::default();

        let done = producer.shape()[..].iter().product::<usize>() == 0;

        // Combine each dimension with its outer dimension if the split mask is not set.
        if P::Dim::RANK > 0 {
            let shape = producer.shape();
            let mut accum = shape[P::Dim::RANK - 1];

            for i in 1..P::Dim::RANK {
                if (P::SPLIT_MASK >> (P::Dim::RANK - 1 - i)) & 1 > 0 {
                    limit[P::Dim::RANK - i] = accum;
                    accum = 1;
                }

                accum *= shape[P::Dim::RANK - 1 - i];
            }

            limit[0] = accum;

            // Set innermost index to simplify checking in Iterator::next.
            if done {
                index[0] = accum;
            }
        }

        Self { producer, limit, index, done }
    }

    unsafe fn step_outer(&mut self) -> bool {
        for i in 1..P::Dim::RANK {
            if (P::SPLIT_MASK >> (i - 1)) & 1 > 0 {
                if self.index[i] + 1 < self.limit[i] {
                    self.producer.step_dim(i);
                    self.index[i] += 1;

                    return true;
                } else {
                    self.producer.reset_dim(i, self.index[i]);
                    self.index[i] = 0;
                }
            }
        }

        false
    }
}

impl<P: Producer + Debug> Debug for Iter<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let is_empty = self.producer.shape()[..].iter().product::<usize>() == 0;
        let is_default = self.index[..] == <P::Dim as Dim>::Shape::default()[..];

        assert!(is_empty || (is_default && !self.done), "iterator in use");

        f.debug_tuple("Iter").field(&self.producer).finish()
    }
}

impl<P: Producer> ExactSizeIterator for Iter<P> {}
impl<P: Producer> FusedIterator for Iter<P> {}

impl<P: Producer> Iterator for Iter<P> {
    type Item = P::Item;

    fn fold<T, F: FnMut(T, Self::Item) -> T>(mut self, init: T, mut f: F) -> T {
        if !self.done {
            if P::Dim::RANK > 0 {
                unsafe { fold::<T, P, <P::Dim as Dim>::Lower>(&mut self, init, &mut f) }
            } else {
                f(init, unsafe { self.producer.get_unchecked(0) })
            }
        } else {
            init
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        if P::Dim::RANK > 0 {
            if self.index[0] == self.limit[0] {
                if self.done || unsafe { !self.step_outer() } {
                    self.done = true;

                    return None;
                }

                self.index[0] = 0;
            }

            self.index[0] += 1;

            unsafe { Some(self.producer.get_unchecked(self.index[0] - 1)) }
        } else if !self.done {
            self.done = true;

            unsafe { Some(self.producer.get_unchecked(0)) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut len = 1;

        for i in 1..P::Dim::RANK {
            if (P::SPLIT_MASK >> (P::Dim::RANK - 1 - i)) & 1 > 0 {
                len = len * self.limit[P::Dim::RANK - i] - self.index[P::Dim::RANK - i];
            }
        }

        len = len * self.limit[0] - self.index[0];

        (len, Some(len))
    }
}

unsafe fn fold<T, P: Producer, I: Dim>(
    iter: &mut Iter<P>,
    mut accum: T,
    f: &mut impl FnMut(T, P::Item) -> T,
) -> T {
    if I::RANK > 0 {
        if P::SPLIT_MASK & (1 << (I::RANK - 1)) > 0 {
            loop {
                accum = fold::<T, P, I::Lower>(iter, accum, f);

                if iter.index[I::RANK] + 1 < iter.limit[I::RANK] {
                    iter.producer.step_dim(I::RANK);
                    iter.index[I::RANK] += 1;
                } else {
                    break;
                }
            }

            iter.producer.reset_dim(I::RANK, iter.index[I::RANK]);
            iter.index[I::RANK] = 0;
        } else {
            accum = fold::<T, P, I::Lower>(iter, accum, f);
        }
    } else {
        for i in iter.index[0]..iter.limit[0] {
            accum = f(accum, iter.producer.get_unchecked(i));
        }

        iter.index[0] = 0;
    }

    accum
}
