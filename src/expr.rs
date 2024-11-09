mod adapters;
mod buffer;
mod expression;
mod into_expr;
mod iter;
mod sources;

pub use adapters::{cloned, copied, enumerate, map, zip, Cloned, Copied, Enumerate, Map, Zip};
pub use buffer::{Buffer, Drain};
pub use expression::{Apply, Expression, FromExpression, IntoExpression};
pub use into_expr::IntoExpr;
pub use iter::Iter;
pub use sources::{fill, fill_with, from_elem, from_fn, Fill, FillWith, FromElem, FromFn};
pub use sources::{AxisExpr, AxisExprMut, Lanes, LanesMut};

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
pub fn fold<T, I: IntoExpression, F: FnMut(T, I::Item) -> T>(expr: I, init: T, f: F) -> T {
    expr.into_expr().fold(init, f)
}

/// Calls a closure on each element of the argument.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, tensor, view};
///
/// let mut t = tensor![0, 1, 2];
///
/// expr::for_each(&mut t, |x| *x *= 2);
///
/// assert_eq!(t, view![0, 2, 4]);
/// ```
pub fn for_each<I: IntoExpression, F: FnMut(I::Item)>(expr: I, f: F) {
    expr.into_expr().for_each(f)
}
