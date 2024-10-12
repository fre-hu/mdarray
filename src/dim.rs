use std::fmt::{Debug, Formatter, Result};
use std::hash::Hash;

/// Array dimension trait.
pub trait Dim: Copy + Debug + Default + Eq + Hash + Send + Sync {
    /// Merge dimensions, where constant size is preferred over dynamic.
    type Merge<D: Dim>: Dim;

    /// Dimension size if known statically, or `None` if dynamic.
    const SIZE: Option<usize>;

    /// Creates an array dimension with the given size.
    ///
    /// # Panics
    ///
    /// Panics if the size is not matching a constant-sized dimension.
    fn from_size(size: usize) -> Self;

    /// Returns the number of elements in the dimension.
    fn size(self) -> usize;
}

#[allow(unreachable_pub)]
pub trait Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync>:
    AsMut<[T]>
    + AsRef<[T]>
    + Clone
    + Debug
    + Default
    + Eq
    + Hash
    + Send
    + Sync
    + for<'a> TryFrom<&'a [T], Error: Debug>
{
    fn new(len: usize) -> Self;
}

/// Type-level constant.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct Const<const N: usize>;

/// Dynamically-sized dimension type.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Dyn(pub usize);

impl<const N: usize> Debug for Const<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Const").field(&N).finish()
    }
}

impl<const N: usize> Dim for Const<N> {
    type Merge<D: Dim> = Self;

    const SIZE: Option<usize> = Some(N);

    fn from_size(size: usize) -> Self {
        assert!(size == N, "invalid size");

        Self
    }

    fn size(self) -> usize {
        N
    }
}

impl Dim for Dyn {
    type Merge<D: Dim> = D;

    const SIZE: Option<usize> = None;

    fn from_size(size: usize) -> Self {
        Self(size)
    }

    fn size(self) -> usize {
        self.0
    }
}

macro_rules! impl_dims {
    ($($n:tt),+) => {
        $(
            impl<T: Copy + Debug + Default + Eq + Hash + Send + Sync> Dims<T> for [T; $n] {
                fn new(len: usize) -> Self {
                    assert!(len == $n, "invalid length");

                    Self::default()
                }
            }
        )+
    };
}

impl_dims!(0, 1, 2, 3, 4, 5, 6);

impl<T: Copy + Debug + Default + Eq + Hash + Send + Sync> Dims<T> for Box<[T]> {
    fn new(len: usize) -> Self {
        vec![T::default(); len].into()
    }
}
