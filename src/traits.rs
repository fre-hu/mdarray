/// Trait for generalization of `Clone` that can reuse an existing object.
pub trait IntoCloned<T> {
    /// Moves an existing object or clones from a reference to the target object.
    fn clone_to(self, target: &mut T);

    /// Returns an existing object or a new clone from a reference.
    fn into_cloned(self) -> T;
}

impl<T: Clone> IntoCloned<T> for &T {
    #[inline]
    fn clone_to(self, target: &mut T) {
        target.clone_from(self);
    }

    #[inline]
    fn into_cloned(self) -> T {
        self.clone()
    }
}

impl<T> IntoCloned<T> for T {
    #[inline]
    fn clone_to(self, target: &mut T) {
        *target = self;
    }

    #[inline]
    fn into_cloned(self) -> T {
        self
    }
}
