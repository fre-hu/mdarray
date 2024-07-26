use std::fmt::{Debug, Formatter, Result};

use crate::dim::Dim;
use crate::expression::Expression;
use crate::iter::Iter;
use crate::traits::IntoExpression;

/// Expression that clones the elements of an underlying expression.
#[derive(Clone, Copy, Debug)]
pub struct Cloned<E> {
    expr: E,
}

/// Expression that copies the elements of an underlying expression.
#[derive(Clone, Copy, Debug)]
pub struct Copied<E> {
    expr: E,
}

/// Expression that gives the current index and the element during iteration.
#[derive(Clone, Copy)]
pub struct Enumerate<E, S> {
    expr: E,
    index: S,
}

/// Expression that calls a closure on each element.
#[derive(Clone, Copy)]
pub struct Map<E, F> {
    expr: E,
    f: F,
}

/// Expression that gives tuples `(x, y)` of the elements from each expression.
#[derive(Clone, Copy, Debug)]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

/// Creates an expression that clones the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression};
///
/// let v = expr![0, 1, 2];
///
/// assert_eq!(expr::cloned(&v).eval(), v);
/// ```
pub fn cloned<'a, T: 'a + Clone, I: IntoExpression<Item = &'a T>>(expr: I) -> Cloned<I::IntoExpr> {
    expr.into_expr().cloned()
}

/// Creates an expression that copies the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression};
///
/// let v = expr![0, 1, 2];
///
/// assert_eq!(expr::copied(&v).eval(), v);
/// ```
pub fn copied<'a, T: 'a + Copy, I: IntoExpression<Item = &'a T>>(expr: I) -> Copied<I::IntoExpr> {
    expr.into_expr().copied()
}

/// Creates an expression that enumerates the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid, Expression};
///
/// let g = grid![1, 2, 3];
///
/// assert_eq!(expr::enumerate(g).eval(), expr![([0], 1), ([1], 2), ([2], 3)]);
/// ```
pub fn enumerate<I: IntoExpression>(expr: I) -> Enumerate<I::IntoExpr, <I::Dim as Dim>::Shape> {
    expr.into_expr().enumerate()
}

/// Creates an expression that calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression};
///
/// let v = expr![0, 1, 2];
///
/// assert_eq!(expr::map(v, |x| 2 * x).eval(), expr![0, 2, 4]);
/// ```
pub fn map<T, I: IntoExpression, F: FnMut(I::Item) -> T>(expr: I, f: F) -> Map<I::IntoExpr, F> {
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
/// use mdarray::{expr, grid, Expression};
///
/// let a = grid![0, 1, 2];
/// let b = grid![3, 4, 5];
///
/// assert_eq!(expr::zip(a, b).eval(), expr![(0, 3), (1, 4), (2, 5)]);
/// ```
pub fn zip<A: IntoExpression, B: IntoExpression>(a: A, b: B) -> Zip<A::IntoExpr, B::IntoExpr> {
    a.into_expr().zip(b)
}

impl<E> Cloned<E> {
    pub(crate) fn new(expr: E) -> Self {
        Self { expr }
    }
}

impl<'a, T: 'a + Clone, E: Expression<Item = &'a T>> Expression for Cloned<E> {
    type Dim = E::Dim;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> <E::Dim as Dim>::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        self.expr.get_unchecked(index).clone()
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.expr.reset_dim(dim, count);
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.expr.step_dim(dim);
    }
}

impl<'a, T: 'a + Clone, E: Expression<Item = &'a T>> IntoIterator for Cloned<E> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E> Copied<E> {
    pub(crate) fn new(expr: E) -> Self {
        Self { expr }
    }
}

impl<'a, T: 'a + Copy, E: Expression<Item = &'a T>> Expression for Copied<E> {
    type Dim = E::Dim;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> <E::Dim as Dim>::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        *self.expr.get_unchecked(index)
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.expr.reset_dim(dim, count);
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.expr.step_dim(dim);
    }
}

impl<'a, T: 'a + Copy, E: Expression<Item = &'a T>> IntoIterator for Copied<E> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E: Expression> Enumerate<E, <E::Dim as Dim>::Shape> {
    pub(crate) fn new(expr: E) -> Self {
        Self { expr, index: Default::default() }
    }
}

impl<E: Debug, S> Debug for Enumerate<E, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Enumerate").field("expr", &self.expr).finish()
    }
}

impl<E: Expression> Expression for Enumerate<E, <E::Dim as Dim>::Shape> {
    type Dim = E::Dim;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = (1 << E::Dim::RANK) - 1;

    fn shape(&self) -> <E::Dim as Dim>::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        if E::Dim::RANK > 0 {
            self.index[0] = index;
        }

        (self.index, self.expr.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.expr.reset_dim(dim, count);
        self.index[dim] = 0;
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.expr.step_dim(dim);
        self.index[dim] += 1;
    }
}

impl<E: Expression> IntoIterator for Enumerate<E, <E::Dim as Dim>::Shape> {
    type Item = (<E::Dim as Dim>::Shape, E::Item);
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E, F> Map<E, F> {
    pub(crate) fn new(expr: E, f: F) -> Self {
        Self { expr, f }
    }
}

impl<E: Debug, F> Debug for Map<E, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Map").field("expr", &self.expr).finish()
    }
}

impl<T, E: Expression, F: FnMut(E::Item) -> T> Expression for Map<E, F> {
    type Dim = E::Dim;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> <E::Dim as Dim>::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        (self.f)(self.expr.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
        self.expr.reset_dim(dim, count);
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.expr.step_dim(dim);
    }
}

impl<T, E: Expression, F: FnMut(E::Item) -> T> IntoIterator for Map<E, F> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<A: Expression, B: Expression> Zip<A, B> {
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

impl<A: Expression, B: Expression> Expression for Zip<A, B> {
    type Dim = <A::Dim as Dim>::Max<B::Dim>;

    const IS_REPEATABLE: bool = A::IS_REPEATABLE && B::IS_REPEATABLE;
    const SPLIT_MASK: usize = A::SPLIT_MASK | B::SPLIT_MASK;

    fn shape(&self) -> <Self::Dim as Dim>::Shape {
        let mut shape = <Self::Dim as Dim>::Shape::default();

        if A::Dim::RANK < B::Dim::RANK {
            shape[..].copy_from_slice(&self.b.shape()[..]);
        } else {
            shape[..].copy_from_slice(&self.a.shape()[..]);
        }

        shape
    }

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

    unsafe fn step_dim(&mut self, dim: usize) {
        if dim < A::Dim::RANK {
            self.a.step_dim(dim);
        }

        if dim < B::Dim::RANK {
            self.b.step_dim(dim);
        }
    }
}

impl<A: Expression, B: Expression> IntoIterator for Zip<A, B> {
    type Item = (A::Item, B::Item);
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}
