use std::fmt::{Debug, Formatter, Result};

use crate::dim::Dim;
use crate::expr::Producer;
use crate::expression::Expression;
use crate::traits::IntoExpression;

/// Expression that clones the elements of an underlying expression.
#[derive(Clone)]
pub struct Cloned<P> {
    producer: P,
}

/// Expression that copies the elements of an underlying expression.
#[derive(Clone)]
pub struct Copied<P> {
    producer: P,
}

/// Expression that gives the current index and the element during iteration.
#[derive(Clone)]
pub struct Enumerate<P, S> {
    producer: P,
    index: S,
}

/// Expression that calls a closure on each element.
#[derive(Clone)]
pub struct Map<P, F> {
    producer: P,
    f: F,
}

/// Expression that gives tuples `(x, y)` of the elements from each expression.
#[derive(Clone)]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

/// Creates an expression that clones the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view, View};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::cloned(&v).eval(), v);
/// ```
pub fn cloned<'a, T: 'a + Clone, I>(expr: I) -> Expression<Cloned<I::Producer>>
where
    I: IntoExpression<Item = &'a T>,
{
    expr.into_expr().cloned()
}

/// Creates an expression that copies the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view, View};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::copied(&v).eval(), v);
/// ```
pub fn copied<'a, T: 'a + Copy, I>(expr: I) -> Expression<Copied<I::Producer>>
where
    I: IntoExpression<Item = &'a T>,
{
    expr.into_expr().copied()
}

/// Creates an expression that enumerates the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid, view, Grid, View};
///
/// let g = grid![1, 2, 3];
///
/// assert_eq!(expr::enumerate(g).eval(), view![([0], 1), ([1], 2), ([2], 3)]);
/// ```
pub fn enumerate<I>(expr: I) -> Expression<Enumerate<I::Producer, <I::Dim as Dim>::Shape>>
where
    I: IntoExpression,
{
    expr.into_expr().enumerate()
}

/// Creates an expression that calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view, View};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::map(v, |x| 2 * x).eval(), view![0, 2, 4]);
/// ```
pub fn map<T, I: IntoExpression, F>(expr: I, f: F) -> Expression<Map<I::Producer, F>>
where
    F: FnMut(I::Item) -> T,
{
    expr.into_expr().map(f)
}

/// Converts the arguments to expressions and zips them.
///
/// # Panics
///
/// Panics if the expressions cannot be broadcast to a common shape.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid, view, Grid, View};
///
/// let a = grid![0, 1, 2];
/// let b = grid![3, 4, 5];
///
/// assert_eq!(expr::zip(a, b).eval(), view![(0, 3), (1, 4), (2, 5)]);
/// ```
pub fn zip<A, B>(a: A, b: B) -> Expression<Zip<A::Producer, B::Producer>>
where
    A: IntoExpression,
    B: IntoExpression,
{
    a.into_expr().zip(b)
}

impl<P> Cloned<P> {
    pub(crate) fn new(producer: P) -> Self {
        Self { producer }
    }
}

impl<P: Producer + Debug> Debug for Cloned<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Cloned").field(&self.producer).finish()
    }
}

impl<'a, T: 'a + Clone, P: Producer<Item = &'a T>> Producer for Cloned<P> {
    type Item = T;
    type Dim = P::Dim;

    const IS_REPEATABLE: bool = P::IS_REPEATABLE;
    const SPLIT_MASK: usize = P::SPLIT_MASK;

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        self.producer.get_unchecked(index).clone()
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.producer.reset_dim(dim, count);
    }

    fn shape(&self) -> <P::Dim as Dim>::Shape {
        self.producer.shape()
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.producer.step_dim(dim);
    }
}

impl<P> Copied<P> {
    pub(crate) fn new(producer: P) -> Self {
        Self { producer }
    }
}

impl<P: Producer + Debug> Debug for Copied<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Copied").field(&self.producer).finish()
    }
}

impl<'a, T: 'a + Copy, P: Producer<Item = &'a T>> Producer for Copied<P> {
    type Item = T;
    type Dim = P::Dim;

    const IS_REPEATABLE: bool = P::IS_REPEATABLE;
    const SPLIT_MASK: usize = P::SPLIT_MASK;

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        *self.producer.get_unchecked(index)
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.producer.reset_dim(dim, count);
    }

    fn shape(&self) -> <P::Dim as Dim>::Shape {
        self.producer.shape()
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.producer.step_dim(dim);
    }
}

impl<P: Producer> Enumerate<P, <P::Dim as Dim>::Shape> {
    pub(crate) fn new(producer: P) -> Self {
        Self { producer, index: Default::default() }
    }
}

impl<P: Debug, S> Debug for Enumerate<P, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Enumerate").field(&self.producer).finish()
    }
}

impl<P: Producer> Producer for Enumerate<P, <P::Dim as Dim>::Shape> {
    type Item = (<P::Dim as Dim>::Shape, P::Item);
    type Dim = P::Dim;

    const IS_REPEATABLE: bool = P::IS_REPEATABLE;
    const SPLIT_MASK: usize = (1 << P::Dim::RANK) - 1;

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        if P::Dim::RANK > 0 {
            self.index[0] = index;
        }

        (self.index, self.producer.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.producer.reset_dim(dim, count);
        self.index[dim] = 0;
    }

    fn shape(&self) -> <P::Dim as Dim>::Shape {
        self.producer.shape()
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.producer.step_dim(dim);
        self.index[dim] += 1;
    }
}

impl<P, F> Map<P, F> {
    pub(crate) fn new(producer: P, f: F) -> Self {
        Self { producer, f }
    }
}

impl<P: Debug, F> Debug for Map<P, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Map").field(&self.producer).finish()
    }
}

impl<T, P: Producer, F: FnMut(P::Item) -> T> Producer for Map<P, F> {
    type Item = T;
    type Dim = P::Dim;

    const IS_REPEATABLE: bool = P::IS_REPEATABLE;
    const SPLIT_MASK: usize = P::SPLIT_MASK;

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        (self.f)(self.producer.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.producer.reset_dim(dim, count);
    }

    fn shape(&self) -> <P::Dim as Dim>::Shape {
        self.producer.shape()
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.producer.step_dim(dim);
    }
}

impl<A: Producer, B: Producer> Zip<A, B> {
    pub(crate) fn new(a: A, b: B) -> Self {
        let min_rank = A::Dim::RANK.min(B::Dim::RANK);

        assert!(a.shape()[..min_rank] == b.shape()[..min_rank], "inner dimensions mismatch");

        let a_len = a.shape()[..].iter().product::<usize>();
        let b_len = b.shape()[..].iter().product::<usize>();

        if A::Dim::RANK < B::Dim::RANK {
            assert!(A::IS_REPEATABLE || a_len == b_len, "expression not repeatable");
        }

        if A::Dim::RANK > B::Dim::RANK {
            assert!(B::IS_REPEATABLE || a_len == b_len, "expression not repeatable");
        }

        Self { a, b }
    }
}

impl<A: Producer + Debug, B: Producer + Debug> Debug for Zip<A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Zip").field(&self.a).field(&self.b).finish()
    }
}

impl<A: Producer, B: Producer> Producer for Zip<A, B> {
    type Item = (A::Item, B::Item);
    type Dim = <A::Dim as Dim>::Max<B::Dim>;

    const IS_REPEATABLE: bool = A::IS_REPEATABLE && B::IS_REPEATABLE;
    const SPLIT_MASK: usize = A::SPLIT_MASK | B::SPLIT_MASK;

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        (self.a.get_unchecked(index), self.b.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        if dim < A::Dim::RANK {
            self.a.reset_dim(dim, count);
        }

        if dim < B::Dim::RANK {
            self.b.reset_dim(dim, count);
        }
    }

    fn shape(&self) -> <Self::Dim as Dim>::Shape {
        let mut shape = <Self::Dim as Dim>::Shape::default();

        if A::Dim::RANK < B::Dim::RANK {
            shape[..].copy_from_slice(&self.b.shape()[..]);
        } else {
            shape[..].copy_from_slice(&self.a.shape()[..]);
        }

        shape
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        if dim < A::Dim::RANK {
            self.a.step_dim(dim);
        }

        if dim < B::Dim::RANK {
            self.b.step_dim(dim);
        }
    }
}
