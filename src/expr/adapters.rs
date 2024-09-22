use std::fmt::{Debug, Formatter, Result};

use crate::expression::Expression;
use crate::iter::Iter;
use crate::shape::Shape;
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
pub struct Enumerate<E, I> {
    expr: E,
    index: I,
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
/// use mdarray::{expr, Expression, FromExpression, Grid};
///
/// let g = Grid::from_expr(expr::cloned(expr![0, 1, 2]));
///
/// assert_eq!(g, expr![0, 1, 2]);
/// ```
pub fn cloned<'a, T: 'a + Clone, I: IntoExpression<Item = &'a T>>(expr: I) -> Cloned<I::IntoExpr> {
    expr.into_expr().cloned()
}

/// Creates an expression that copies the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression, FromExpression, Grid};
///
/// let g = Grid::from_expr(expr::copied(expr![0, 1, 2]));
///
/// assert_eq!(g, expr![0, 1, 2]);
/// ```
pub fn copied<'a, T: 'a + Copy, I: IntoExpression<Item = &'a T>>(expr: I) -> Copied<I::IntoExpr> {
    expr.into_expr().copied()
}

/// Creates an expression that enumerates the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid, Expression, FromExpression, Grid};
///
/// let g = Grid::from_expr(expr::enumerate(grid![1, 2, 3]));
///
/// assert_eq!(g, expr![([0], 1), ([1], 2), ([2], 3)]);
/// ```
pub fn enumerate<I: IntoExpression>(expr: I) -> Enumerate<I::IntoExpr, <I::Shape as Shape>::Dims> {
    expr.into_expr().enumerate()
}

/// Creates an expression that calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression, FromExpression, Grid};
///
/// let g = Grid::from_expr(expr::map(expr![0, 1, 2], |x| 2 * x));
///
/// assert_eq!(g, expr![0, 2, 4]);
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
/// use mdarray::{expr, grid, Expression, FromExpression, Grid};
///
/// let a = grid![0, 1, 2];
/// let b = grid![3, 4, 5];
///
/// let g = Grid::from_expr(expr::zip(a, b));
///
/// assert_eq!(g, expr![(0, 3), (1, 4), (2, 5)]);
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
    type Shape = E::Shape;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        self.expr.get_unchecked(index).clone()
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        self.expr.reset_dim(index, count);
    }

    unsafe fn step_dim(&mut self, index: usize) {
        self.expr.step_dim(index);
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
    type Shape = E::Shape;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        *self.expr.get_unchecked(index)
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        self.expr.reset_dim(index, count);
    }

    unsafe fn step_dim(&mut self, index: usize) {
        self.expr.step_dim(index);
    }
}

impl<'a, T: 'a + Copy, E: Expression<Item = &'a T>> IntoIterator for Copied<E> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E: Expression> Enumerate<E, <E::Shape as Shape>::Dims> {
    pub(crate) fn new(expr: E) -> Self {
        Self { expr, index: Default::default() }
    }
}

impl<E: Debug, I> Debug for Enumerate<E, I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Enumerate").field("expr", &self.expr).finish()
    }
}

impl<E: Expression> Expression for Enumerate<E, <E::Shape as Shape>::Dims> {
    type Shape = E::Shape;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = (1 << E::Shape::RANK) - 1;

    fn shape(&self) -> E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        if E::Shape::RANK > 0 {
            self.index[0] = index;
        }

        (self.index, self.expr.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        self.expr.reset_dim(index, count);
        self.index[index] = 0;
    }

    unsafe fn step_dim(&mut self, index: usize) {
        self.expr.step_dim(index);
        self.index[index] += 1;
    }
}

impl<E: Expression> IntoIterator for Enumerate<E, <E::Shape as Shape>::Dims> {
    type Item = (<E::Shape as Shape>::Dims, E::Item);
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
    type Shape = E::Shape;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;
    const SPLIT_MASK: usize = E::SPLIT_MASK;

    fn shape(&self) -> E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        (self.f)(self.expr.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        self.expr.reset_dim(index, count);
    }

    unsafe fn step_dim(&mut self, index: usize) {
        self.expr.step_dim(index);
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
        assert!(A::IS_REPEATABLE || A::Shape::RANK >= B::Shape::RANK, "expression not repeatable");
        assert!(B::IS_REPEATABLE || B::Shape::RANK >= A::Shape::RANK, "expression not repeatable");

        let min_rank = A::Shape::RANK.min(B::Shape::RANK);

        assert!(a.dims()[..min_rank] == b.dims()[..min_rank], "inner dimensions mismatch");

        Self { a, b }
    }
}

impl<A: Expression, B: Expression> Expression for Zip<A, B> {
    type Shape = <A::Shape as Shape>::Merge<B::Shape>;

    const IS_REPEATABLE: bool = A::IS_REPEATABLE && B::IS_REPEATABLE;
    const SPLIT_MASK: usize = A::SPLIT_MASK | B::SPLIT_MASK;

    fn shape(&self) -> Self::Shape {
        let mut dims = <Self::Shape as Shape>::Dims::default();

        if A::Shape::RANK < B::Shape::RANK {
            dims[..].copy_from_slice(&self.b.dims()[..]);
        } else {
            dims[..].copy_from_slice(&self.a.dims()[..]);
        }

        Shape::from_dims(dims)
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        (self.a.get_unchecked(index), self.b.get_unchecked(index))
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        if index < A::Shape::RANK {
            self.a.reset_dim(index, count);
        }

        if index < B::Shape::RANK {
            self.b.reset_dim(index, count);
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        if index < A::Shape::RANK {
            self.a.step_dim(index);
        }

        if index < B::Shape::RANK {
            self.b.step_dim(index);
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
