use crate::dim::Dim;
use crate::traits::IntoExpression;

mod adapters;
mod sources;

pub use adapters::{cloned, copied, enumerate, map, zip, Cloned, Copied, Enumerate, Map, Zip};
pub use sources::{fill, fill_with, from_elem, from_fn, Fill, FillWith, FromElem, FromFn};
pub use sources::{AxisExpr, AxisExprMut, Drain, Expr, ExprMut, IntoExpr};

/// Expression producer trait.
pub trait Producer {
    /// Element type from the expression.
    type Item;

    /// Array dimension type.
    type Dim: Dim;

    /// True if the expression can be restarted from the beginning after the last index.
    const IS_REPEATABLE: bool;

    /// Bitmask per dimension indicating if indexing can be combined with the outer dimension.
    const SPLIT_MASK: usize;

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item;

    #[doc(hidden)]
    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}

    #[doc(hidden)]
    fn shape(&self) -> <Self::Dim as Dim>::Shape;

    #[doc(hidden)]
    unsafe fn step_dim(&mut self, _: usize) {}
}

/// Folds all elements of the argument into an accumulator by applying an operation,
/// and returns the result.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view, View};
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
/// use mdarray::{expr, grid, view, Grid, View};
///
/// let mut g = grid![0, 1, 2];
///
/// expr::for_each(&mut g, |x| *x *= 2);
///
/// assert_eq!(g, view![0, 2, 4]);
/// ```
pub fn for_each<I: IntoExpression, F: FnMut(I::Item)>(expr: I, f: F) {
    expr.into_expr().for_each(f)
}
