use crate::expression::Expression;
use crate::traits::IntoExpression;

mod adapters;
#[allow(clippy::module_inception)]
mod expr;
mod into_expr;
mod sources;

pub use adapters::{cloned, copied, enumerate, map, zip, Cloned, Copied, Enumerate, Map, Zip};
pub use expr::{Expr, ExprMut};
pub use into_expr::IntoExpr;
pub use sources::{fill, fill_with, from_elem, from_fn, Fill, FillWith, FromElem, FromFn};
pub use sources::{AxisExpr, AxisExprMut, Lanes, LanesMut};

/// Folds all elements of the argument into an accumulator by applying an operation,
/// and returns the result.
///
/// # Examples
///
/// ```
/// use mdarray::expr;
///
/// let v = expr![0, 1, 2];
///
/// assert_eq!(expr::fold(v, 0, |acc, x| acc + x), 3);
/// ```
pub fn fold<T, I: IntoExpression, F: FnMut(T, I::Item) -> T>(expr: I, init: T, f: F) -> T {
    expr.into_expr().fold(init, f)
}

/// Calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid};
///
/// let mut g = grid![0, 1, 2];
///
/// expr::for_each(&mut g, |x| *x *= 2);
///
/// assert_eq!(g, expr![0, 2, 4]);
/// ```
pub fn for_each<I: IntoExpression, F: FnMut(I::Item)>(expr: I, f: F) {
    expr.into_expr().for_each(f)
}
