use std::borrow::Borrow;

use crate::dim::Const;
use crate::expr::FromExpression;
use crate::shape::Shape;
use crate::slice::Slice;

/// Trait for generalization of `Clone` that can reuse an existing object.
pub trait IntoCloned<T> {
    /// Moves an existing object or clones from a reference to the target object.
    fn clone_to(self, target: &mut T);

    /// Returns an existing object or a new clone from a reference.
    fn into_cloned(self) -> T;
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

/// Trait for a multidimensional array owning its contents.
pub trait Owned<T, S: Shape>: Borrow<Slice<T, S>> + FromExpression<T, S> {
    #[doc(hidden)]
    type WithConst<const N: usize>: Owned<T, S::Prepend<Const<N>>>;

    #[doc(hidden)]
    fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone;
}
