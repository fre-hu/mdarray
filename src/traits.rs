use crate::dim::Dim;
use crate::expr::Producer;
use crate::expression::Expression;

/// Trait for applying a closure and returning an existing array or an expression.
pub trait Apply<T>: IntoExpression {
    /// The resulting type after applying a closure.
    type Output<F: FnMut(Self::Item) -> T>: IntoExpression<Item = T, Dim = Self::Dim>;

    /// The resulting type after zipping elements and applying a closure.
    type ZippedWith<I: IntoExpression, F: FnMut(Self::Item, I::Item) -> T>: IntoExpression<Item = T>;

    /// Returns the array or an expression with the given closure applied to each element.
    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output<F>;

    /// Returns the array or an expression with the given closure applied to zipped element pairs.
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut(Self::Item, I::Item) -> T;
}

/// Trait for generalization of `Clone` that can reuse an existing object.
pub trait IntoCloned<T> {
    /// Moves an existing object or clones from a reference to the target object.
    fn clone_to(self, target: &mut T);

    /// Returns an existing object or a new clone from a reference.
    fn into_cloned(self) -> T;
}

/// Conversion trait into an expression.
pub trait IntoExpression {
    /// Array element type.
    type Item;

    /// Array dimension type.
    type Dim: Dim;

    /// Which kind of expression producer are we turning this into?
    type Producer: Producer<Item = Self::Item, Dim = Self::Dim>;

    /// Creates an expression from a value.
    fn into_expr(self) -> Expression<Self::Producer>;
}

impl<T: Clone> IntoCloned<T> for &T {
    fn clone_to(self, target: &mut T) {
        target.clone_from(self);
    }

    fn into_cloned(self) -> T {
        self.clone()
    }
}

impl<T> IntoCloned<T> for T {
    fn clone_to(self, target: &mut T) {
        *target = self;
    }

    fn into_cloned(self) -> T {
        self
    }
}
