use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::traits::Owned;

/// Array dimension trait.
pub trait Dim: Copy + Debug + Default + Hash + Ord + Send + Sync {
    /// Merge dimensions, where constant size is preferred over dynamic.
    type Merge<D: Dim>: Dim;

    #[doc(hidden)]
    type Owned<T, S: Shape>: Owned<T, S::Prepend<Self>>;

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
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Const<const N: usize>;

/// Dynamically-sized dimension type.
pub type Dyn = usize;

impl<const N: usize> Debug for Const<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Const").field(&N).finish()
    }
}

impl<const N: usize> Dim for Const<N> {
    type Merge<D: Dim> = Self;
    type Owned<T, S: Shape> = <S::Owned<T> as Owned<T, S>>::WithConst<N>;

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
    type Owned<T, S: Shape> = Tensor<T, S::Prepend<Self>>;

    const SIZE: Option<usize> = None;

    fn from_size(size: usize) -> Self {
        size
    }

    fn size(self) -> usize {
        self
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

impl<const N: usize> From<Const<N>> for Dyn {
    fn from(_: Const<N>) -> Self {
        N
    }
}

impl<const N: usize> TryFrom<Dyn> for Const<N> {
    type Error = Dyn;

    fn try_from(value: Dyn) -> Result<Self, Self::Error> {
        if value.size() == N { Ok(Self) } else { Err(value) }
    }
}
