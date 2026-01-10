#[cfg(feature = "nightly")]
use alloc::alloc::Allocator;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec;

use core::fmt::{self, Debug, Formatter};
use core::hash::Hash;

#[cfg(not(feature = "nightly"))]
use crate::allocator::Allocator;
use crate::buffer::{DynBuffer, Owned};
use crate::shape::Shape;

/// Array dimension trait.
pub trait Dim: Copy + Debug + Default + Hash + Ord + Send + Sync {
    /// Merge dimensions, where constant size is preferred over dynamic.
    type Merge<D: Dim>: Dim;

    #[doc(hidden)]
    type Buffer<T, S: Shape, A: Allocator>: Owned<Item = T, Shape = S::Prepend<Self>, Alloc = A>;

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
pub trait Dims<T: Copy + Debug + Default + Hash + Ord + Send + Sync>:
    AsMut<[T]>
    + AsRef<[T]>
    + Clone
    + Debug
    + Default
    + Hash
    + Ord
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
    type Buffer<T, S: Shape, A: Allocator> = <S::Buffer<T, A> as Owned>::WithConst<N>;

    const SIZE: Option<usize> = Some(N);

    #[inline]
    fn from_size(size: usize) -> Self {
        assert!(size == N, "invalid size");

        Self
    }

    #[inline]
    fn size(self) -> usize {
        N
    }
}

impl Dim for Dyn {
    type Merge<D: Dim> = D;
    type Buffer<T, S: Shape, A: Allocator> = DynBuffer<T, S::Prepend<Self>, A>;

    const SIZE: Option<usize> = None;

    #[inline]
    fn from_size(size: usize) -> Self {
        size
    }

    #[inline]
    fn size(self) -> usize {
        self
    }
}

macro_rules! impl_dims {
    ($($n:tt),+) => {
        $(
            impl<T: Copy + Debug + Default + Hash + Ord + Send + Sync> Dims<T> for [T; $n] {
                #[inline]
                fn new(len: usize) -> Self {
                    assert!(len == $n, "invalid length");

                    Self::default()
                }
            }
        )+
    };
}

impl_dims!(0, 1, 2, 3, 4, 5, 6);

impl<T: Copy + Debug + Default + Hash + Ord + Send + Sync> Dims<T> for Box<[T]> {
    #[inline]
    fn new(len: usize) -> Self {
        vec![T::default(); len].into()
    }
}

impl<const N: usize> From<Const<N>> for Dyn {
    #[inline]
    fn from(_: Const<N>) -> Self {
        N
    }
}

impl<const N: usize> TryFrom<Dyn> for Const<N> {
    type Error = Dyn;

    #[inline]
    fn try_from(value: Dyn) -> Result<Self, Self::Error> {
        if value.size() == N { Ok(Self) } else { Err(value) }
    }
}
