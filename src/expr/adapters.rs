use std::fmt::{Debug, Formatter, Result};

use crate::expr::expression::{Expression, IntoExpression};
use crate::expr::iter::Iter;
use crate::shape::Shape;

/// Expression that clones the elements of an underlying expression.
#[derive(Clone, Debug)]
pub struct Cloned<E> {
    expr: E,
}

/// Expression that copies the elements of an underlying expression.
#[derive(Clone, Debug)]
pub struct Copied<E> {
    expr: E,
}

/// Expression that gives the current count and the element during iteration.
#[derive(Clone)]
pub struct Enumerate<E> {
    expr: E,
    count: usize,
}

/// Expression that calls a closure on each element.
#[derive(Clone)]
pub struct Map<E, F> {
    expr: E,
    f: F,
}

/// Expression that gives tuples `(x, y)` of the elements from each expression.
#[derive(Clone)]
pub struct Zip<A: Expression, B: Expression> {
    a: A,
    b: B,
    shape: <Self as Expression>::Shape,
}

/// Creates an expression that clones the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, expr::Expression, view};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::cloned(v).eval(), v);
/// ```
pub fn cloned<'a, T: 'a + Clone, I: IntoExpression<Item = &'a T>>(expr: I) -> Cloned<I::IntoExpr> {
    expr.into_expr().cloned()
}

/// Creates an expression that copies the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, expr::Expression, view};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::copied(v).eval(), v);
/// ```
pub fn copied<'a, T: 'a + Copy, I: IntoExpression<Item = &'a T>>(expr: I) -> Copied<I::IntoExpr> {
    expr.into_expr().copied()
}

/// Creates an expression that enumerates the elements of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, expr::Expression, tensor, view};
///
/// let t = tensor![3, 4, 5];
///
/// assert_eq!(expr::enumerate(t).eval(), view![(0, 3), (1, 4), (2, 5)]);
/// ```
pub fn enumerate<I: IntoExpression>(expr: I) -> Enumerate<I::IntoExpr> {
    expr.into_expr().enumerate()
}

/// Creates an expression that calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, expr::Expression, view};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::map(v, |x| 2 * x).eval(), view![0, 2, 4]);
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
/// use mdarray::{expr, expr::Expression, tensor, view};
///
/// let a = tensor![0, 1, 2];
/// let b = tensor![3, 4, 5];
///
/// assert_eq!(expr::zip(a, b).eval(), view![(0, 3), (1, 4), (2, 5)]);
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

    fn shape(&self) -> &E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        unsafe { self.expr.get_unchecked(index).clone() }
    }

    fn inner_rank(&self) -> usize {
        self.expr.inner_rank()
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        unsafe {
            self.expr.reset_dim(index, count);
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        unsafe {
            self.expr.step_dim(index);
        }
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

    fn shape(&self) -> &E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        unsafe { *self.expr.get_unchecked(index) }
    }

    fn inner_rank(&self) -> usize {
        self.expr.inner_rank()
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        unsafe {
            self.expr.reset_dim(index, count);
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        unsafe {
            self.expr.step_dim(index);
        }
    }
}

impl<'a, T: 'a + Copy, E: Expression<Item = &'a T>> IntoIterator for Copied<E> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<E: Expression> Enumerate<E> {
    pub(crate) fn new(expr: E) -> Self {
        Self { expr, count: 0 }
    }
}

impl<E: Debug> Debug for Enumerate<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Enumerate").field("expr", &self.expr).finish()
    }
}

impl<E: Expression> Expression for Enumerate<E> {
    type Shape = E::Shape;

    const IS_REPEATABLE: bool = E::IS_REPEATABLE;

    fn shape(&self) -> &E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        self.count += 1;

        unsafe { (self.count - 1, self.expr.get_unchecked(index)) }
    }

    fn inner_rank(&self) -> usize {
        self.expr.inner_rank()
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        unsafe {
            self.expr.reset_dim(index, count);
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        unsafe {
            self.expr.step_dim(index);
        }
    }
}

impl<E: Expression> IntoIterator for Enumerate<E> {
    type Item = (usize, E::Item);
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

    fn shape(&self) -> &E::Shape {
        self.expr.shape()
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        unsafe { (self.f)(self.expr.get_unchecked(index)) }
    }

    fn inner_rank(&self) -> usize {
        self.expr.inner_rank()
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        unsafe {
            self.expr.reset_dim(index, count);
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        unsafe {
            self.expr.step_dim(index);
        }
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
        assert!(A::IS_REPEATABLE || a.rank() >= b.rank(), "expression not repeatable");
        assert!(B::IS_REPEATABLE || b.rank() >= a.rank(), "expression not repeatable");

        let shape = a.shape().with_dims(|a_dims| {
            b.shape().with_dims(|b_dims| {
                let dims = if a_dims.len() < b_dims.len() { b_dims } else { a_dims };
                let inner_match =
                    a_dims[dims.len() - b_dims.len()..] == b_dims[dims.len() - a_dims.len()..];

                assert!(inner_match, "inner dimensions mismatch");

                Shape::from_dims(dims)
            })
        });

        Self { a, b, shape }
    }
}

impl<A: Expression + Debug, B: Expression + Debug> Debug for Zip<A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Zip").field("a", &self.a).field("b", &self.b).finish()
    }
}

impl<S: Shape, R: Shape, A, B> Expression for Zip<A, B>
where
    A: Expression<Shape = S>,
    B: Expression<Shape = R>,
{
    type Shape = <<S::Reverse as Shape>::Merge<R::Reverse> as Shape>::Reverse;

    const IS_REPEATABLE: bool = A::IS_REPEATABLE && B::IS_REPEATABLE;

    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
        unsafe { (self.a.get_unchecked(index), self.b.get_unchecked(index)) }
    }

    fn inner_rank(&self) -> usize {
        self.a.inner_rank().min(self.b.inner_rank())
    }

    unsafe fn reset_dim(&mut self, index: usize, count: usize) {
        let delta = self.shape.rank() - index;

        unsafe {
            if delta <= self.a.rank() {
                self.a.reset_dim(self.a.rank() - delta, count);
            }

            if delta <= self.b.rank() {
                self.b.reset_dim(self.b.rank() - delta, count);
            }
        }
    }

    unsafe fn step_dim(&mut self, index: usize) {
        let delta = self.shape.rank() - index;

        unsafe {
            if delta <= self.a.rank() {
                self.a.step_dim(self.a.rank() - delta);
            }

            if delta <= self.b.rank() {
                self.b.step_dim(self.b.rank() - delta);
            }
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
