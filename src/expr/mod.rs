//! Expression module, for multidimensional iteration.

mod adapters;
mod expression;
mod into_expr;
mod iter;
mod sources;

pub use adapters::{Cloned, Copied, Enumerate, Map, Zip, cloned, copied, enumerate, map, zip};
pub use expression::{Apply, Expand, Expression, FromExpression, IntoExpression};
pub use into_expr::IntoExpr;
pub use iter::Iter;
pub use sources::{AxisExpr, AxisExprMut, Lanes, LanesMut};
pub use sources::{Fill, FillWith, FromElem, FromFn, fill, fill_with, from_elem, from_fn};

/// Folds all elements of the argument into an accumulator by applying an operation,
/// and returns the result.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view};
///
/// let v = view![0, 1, 2];
///
/// assert_eq!(expr::fold(v, 0, |acc, x| acc + x), 3);
/// ```
#[inline]
pub fn fold<T, I: IntoExpression, F: FnMut(T, I::Item) -> T>(expr: I, init: T, f: F) -> T {
    expr.into_expr().fold(init, f)
}

/// Calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{array, expr, view};
///
/// let mut a = array![0, 1, 2];
///
/// expr::for_each(&mut a, |x| *x *= 2);
///
/// assert_eq!(a, view![0, 2, 4]);
/// ```
#[inline]
pub fn for_each<I: IntoExpression, F: FnMut(I::Item)>(expr: I, f: F) {
    expr.into_expr().for_each(f);
}
