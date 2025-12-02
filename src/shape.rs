#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::{Hash, Hasher};
use core::slice;

use crate::array::Array;
use crate::dim::{Const, Dim, Dims, Dyn};
use crate::layout::{Layout, Strided};
use crate::tensor::Tensor;
use crate::traits::Owned;

/// Array shape trait.
pub trait Shape: Clone + Debug + Default + Hash + Ord + Send + Sync {
    /// First dimension.
    type Head: Dim;

    /// Shape excluding the first dimension.
    type Tail: Shape;

    /// Shape with the reverse ordering of dimensions.
    type Reverse: Shape;

    /// Prepend the dimension to the shape.
    type Prepend<D: Dim>: Shape;

    /// Corresponding shape with dynamically-sized dimensions.
    type Dyn: Shape;

    /// Merge each dimension pair, where constant size is preferred over dynamic.
    /// The result has dynamic rank if at least one of the inputs has dynamic rank.
    type Merge<S: Shape>: Shape;

    /// Select layout `L` for rank 0, or `Strided` for rank >0 or dynamic.
    type Layout<L: Layout>: Layout;

    /// Corresponding array type owning its contents.
    type Owned<T>: Owned<T, Self>;

    #[doc(hidden)]
    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync>: Dims<T>;

    /// Array rank if known statically, or `None` if dynamic.
    const RANK: Option<usize>;

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(&self, index: usize) -> usize {
        assert!(index < self.rank(), "invalid dimension");

        self.with_dims(|dims| dims[index])
    }

    /// Creates an array shape with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not matching static rank or constant-sized dimensions.
    fn from_dims(dims: &[usize]) -> Self {
        let mut shape = Self::new(dims.len());

        shape.with_mut_dims(|dst| dst.copy_from_slice(dims));
        shape
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.with_dims(|dims| dims.iter().product())
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        self.with_dims(|dims| dims.len())
    }

    #[doc(hidden)]
    fn new(rank: usize) -> Self;

    #[doc(hidden)]
    fn with_dims<T, F: FnOnce(&[usize]) -> T>(&self, f: F) -> T;

    #[doc(hidden)]
    fn with_mut_dims<T, F: FnOnce(&mut [usize]) -> T>(&mut self, f: F) -> T;

    #[doc(hidden)]
    fn checked_len(&self) -> Option<usize> {
        self.with_dims(|dims| dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x)))
    }

    #[doc(hidden)]
    fn prepend_dim<S: Shape>(&self, size: usize) -> S {
        let mut shape = S::new(self.rank() + 1);

        shape.with_mut_dims(|dims| {
            dims[0] = size;
            self.with_dims(|src| dims[1..].copy_from_slice(src));
        });

        shape
    }

    #[doc(hidden)]
    fn remove_dim<S: Shape>(&self, index: usize) -> S {
        assert!(index < self.rank(), "invalid dimension");

        let mut shape = S::new(self.rank() - 1);

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| {
                dims[..index].copy_from_slice(&src[..index]);
                dims[index..].copy_from_slice(&src[index + 1..]);
            });
        });

        shape
    }

    #[doc(hidden)]
    fn reshape<S: Shape>(&self, mut new_shape: S) -> S {
        let mut inferred = None;

        new_shape.with_mut_dims(|dims| {
            for i in 0..dims.len() {
                if dims[i] == usize::MAX {
                    assert!(inferred.is_none(), "at most one dimension can be inferred");

                    dims[i] = 1;
                    inferred = Some(i);
                }
            }
        });

        let old_len = self.len();
        let new_len = new_shape.checked_len().expect("invalid length");

        if let Some(i) = inferred {
            assert!(old_len % new_len == 0, "length not divisible by the new dimensions");

            new_shape.with_mut_dims(|dims| dims[i] = old_len / new_len);
        } else {
            assert!(new_len == old_len, "length must not change");
        }

        new_shape
    }

    #[doc(hidden)]
    fn resize_dim<S: Shape>(&self, index: usize, new_size: usize) -> S {
        assert!(index < self.rank(), "invalid dimension");

        let mut shape = S::new(self.rank());

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| dims[..].copy_from_slice(src));
            dims[index] = new_size;
        });

        shape
    }

    #[doc(hidden)]
    fn reverse(&self) -> Self::Reverse {
        let mut shape = Self::Reverse::new(self.rank());

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| dims.copy_from_slice(src));
            dims.reverse();
        });

        shape
    }
}

/// Trait for array shape where all dimensions are constant-sized.
pub trait ConstShape: Copy + Shape {
    #[doc(hidden)]
    type Inner<T>;

    #[doc(hidden)]
    type WithConst<T, const N: usize, A: Owned<T, Self>>: Owned<T, Self::Prepend<Const<N>>>;
}

/// Conversion trait into an array shape.
pub trait IntoShape {
    /// Which kind of array shape are we turning this into?
    type IntoShape: Shape;

    /// Creates an array shape from a value.
    fn into_shape(self) -> Self::IntoShape;

    #[doc(hidden)]
    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T;
}

/// Array shape type with dynamic rank.
///
/// If the rank is 0 or 1, no heap allocation is necessary. The default value
/// will have rank 1 and contain no elements.
pub enum DynRank {
    /// Shape variant with dynamic rank.
    Dyn(Box<[usize]>),
    /// Shape variant with rank 1.
    One(usize),
}

/// Array shape type with dynamically-sized dimensions.
pub type Rank<const N: usize> = <[usize; N] as IntoShape>::IntoShape;

impl DynRank {
    /// Returns the number of elements in each dimension.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_ref(size),
        }
    }
}

impl Clone for DynRank {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::One(dim) => Self::One(*dim),
            Self::Dyn(dims) => {
                if dims.len() == 1 {
                    Self::One(dims[0])
                } else {
                    Self::Dyn(dims.clone())
                }
            }
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        if let Self::Dyn(dims) = self {
            if let Self::Dyn(src) = source {
                if dims.len() == src.len() {
                    dims.clone_from_slice(src);

                    return;
                }
            }
        }

        *self = source.clone();
    }
}

impl Debug for DynRank {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.with_dims(|dims| f.debug_tuple("DynRank").field(&dims).finish())
    }
}

impl Default for DynRank {
    #[inline]
    fn default() -> Self {
        Self::One(0)
    }
}

impl Eq for DynRank {}

impl Hash for DynRank {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with_dims(|dims| dims.hash(state))
    }
}

impl Ord for DynRank {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.with_dims(|dims| other.with_dims(|other| dims.cmp(other)))
    }
}

impl PartialEq for DynRank {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.with_dims(|dims| other.with_dims(|other| dims.eq(other)))
    }
}

impl PartialOrd for DynRank {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Shape for DynRank {
    type Head = Dyn;
    type Tail = Self;

    type Reverse = Self;
    type Prepend<D: Dim> = Self;

    type Dyn = Self;
    type Merge<S: Shape> = Self;

    type Layout<L: Layout> = Strided;
    type Owned<T> = Tensor<T>;

    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = Box<[T]>;

    const RANK: Option<usize> = None;

    #[inline]
    fn new(rank: usize) -> Self {
        if rank == 1 { Self::One(0) } else { Self::Dyn(Dims::new(rank)) }
    }

    #[inline]
    fn with_dims<T, F: FnOnce(&[usize]) -> T>(&self, f: F) -> T {
        let dims = match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_ref(size),
        };

        f(dims)
    }

    #[inline]
    fn with_mut_dims<T, F: FnOnce(&mut [usize]) -> T>(&mut self, f: F) -> T {
        let dims = match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_mut(size),
        };

        f(dims)
    }
}

impl Shape for () {
    type Head = Dyn;
    type Tail = Self;

    type Reverse = Self;
    type Prepend<D: Dim> = (D,);

    type Dyn = Self;
    type Merge<S: Shape> = S;

    type Layout<L: Layout> = L;
    type Owned<T> = Array<T, ()>;

    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; 0];

    const RANK: Option<usize> = Some(0);

    #[inline]
    fn new(rank: usize) {
        assert!(rank == 0, "invalid rank");
    }

    #[inline]
    fn with_dims<T, F: FnOnce(&[usize]) -> T>(&self, f: F) -> T {
        f(&[])
    }

    #[inline]
    fn with_mut_dims<T, F: FnOnce(&mut [usize]) -> T>(&mut self, f: F) -> T {
        f(&mut [])
    }
}

impl<X: Dim> Shape for (X,) {
    type Head = X;
    type Tail = ();

    type Reverse = Self;
    type Prepend<D: Dim> = (D, X);

    type Dyn = (Dyn,);
    type Merge<S: Shape> = <S::Tail as Shape>::Prepend<X::Merge<S::Head>>;

    type Layout<L: Layout> = Strided;
    type Owned<T> = X::Owned<T, ()>;

    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; 1];

    const RANK: Option<usize> = Some(1);

    fn new(rank: usize) -> Self {
        assert!(rank == 1, "invalid rank");

        Self::default()
    }

    fn with_dims<T, F: FnOnce(&[usize]) -> T>(&self, f: F) -> T {
        f(&[self.0.size()])
    }

    fn with_mut_dims<T, F: FnOnce(&mut [usize]) -> T>(&mut self, f: F) -> T {
        let mut dims = [self.0.size()];
        let value = f(&mut dims);

        *self = (X::from_size(dims[0]),);

        value
    }
}

#[cfg(not(feature = "nightly"))]
macro_rules! dyn_shape {
    ($($yz:tt),+) => {
        <<Self::Tail as Shape>::Dyn as Shape>::Prepend<Dyn>
    };
}

#[cfg(feature = "nightly")]
macro_rules! dyn_shape {
    ($($yz:tt),+) => {
        (Dyn $(,${ignore($yz)} Dyn)+)
    };
}

macro_rules! impl_shape {
    ($n:tt, ($($jk:tt),+), ($($yz:tt),+), $reverse:tt, $prepend:tt) => {
        impl<X: Dim $(,$yz: Dim)+> Shape for (X $(,$yz)+) {
            type Head = X;
            type Tail = ($($yz,)+);

            type Reverse = $reverse;
            type Prepend<D: Dim> = $prepend;

            type Dyn = dyn_shape!($($yz),+);
            type Merge<S: Shape> =
                <<Self::Tail as Shape>::Merge<S::Tail> as Shape>::Prepend<X::Merge<S::Head>>;

            type Layout<L: Layout> = Strided;
            type Owned<T> = X::Owned<T, Self::Tail>;

            type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; $n];

            const RANK: Option<usize> = Some($n);

            fn new(rank: usize) -> Self {
                assert!(rank == $n, "invalid rank");

                Self::default()
            }

            fn with_dims<T, F: FnOnce(&[usize]) -> T>(&self, f: F) -> T {
                f(&[self.0.size() $(,self.$jk.size())+])
            }

            fn with_mut_dims<T, F: FnOnce(&mut [usize]) -> T>(&mut self, f: F) -> T {
                let mut dims = [self.0.size() $(,self.$jk.size())+];
                let value = f(&mut dims);

                *self = (X::from_size(dims[0]) $(,$yz::from_size(dims[$jk]))+);

                value
            }
        }
    };
}

impl_shape!(2, (1), (Y), (Y, X), (D, X, Y));
impl_shape!(3, (1, 2), (Y, Z), (Z, Y, X), (D, X, Y, Z));
impl_shape!(4, (1, 2, 3), (Y, Z, W), (W, Z, Y, X), (D, X, Y, Z, W));
impl_shape!(5, (1, 2, 3, 4), (Y, Z, W, U), (U, W, Z, Y, X), (D, X, Y, Z, W, U));
impl_shape!(6, (1, 2, 3, 4, 5), (Y, Z, W, U, V), (V, U, W, Z, Y, X), DynRank);

macro_rules! impl_const_shape {
    (($($xyz:tt),*), $inner:ty, $with_const:tt) => {
        impl<$(const $xyz: usize),*> ConstShape for ($(Const<$xyz>,)*) {
            type Inner<T> = $inner;
            type WithConst<T, const N: usize, A: Owned<T, Self>> =
                $with_const<T, Self::Prepend<Const<N>>>;
        }
    };
}

impl_const_shape!((), T, Array);
impl_const_shape!((X), [T; X], Array);
impl_const_shape!((X, Y), [[T; Y]; X], Array);
impl_const_shape!((X, Y, Z), [[[T; Z]; Y]; X], Array);
impl_const_shape!((X, Y, Z, W), [[[[T; W]; Z]; Y]; X], Array);
impl_const_shape!((X, Y, Z, W, U), [[[[[T; U]; W]; Z]; Y]; X], Array);
impl_const_shape!((X, Y, Z, W, U, V), [[[[[[T; V]; U]; W]; Z]; Y]; X], Tensor);

impl<S: Shape> IntoShape for S {
    type IntoShape = S;

    fn into_shape(self) -> S {
        self
    }

    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        self.with_dims(f)
    }
}

impl<const N: usize> IntoShape for &[usize; N] {
    type IntoShape = DynRank;

    fn into_shape(self) -> DynRank {
        Shape::from_dims(self)
    }

    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(self)
    }
}

impl IntoShape for &[usize] {
    type IntoShape = DynRank;

    #[inline]
    fn into_shape(self) -> DynRank {
        Shape::from_dims(self)
    }

    #[inline]
    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(self)
    }
}

impl IntoShape for Box<[usize]> {
    type IntoShape = DynRank;

    #[inline]
    fn into_shape(self) -> DynRank {
        DynRank::Dyn(self)
    }

    #[inline]
    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(&self)
    }
}

impl<const N: usize> IntoShape for Const<N> {
    type IntoShape = (Self,);

    fn into_shape(self) -> Self::IntoShape {
        (self,)
    }

    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(&[N])
    }
}

impl IntoShape for Dyn {
    type IntoShape = (Self,);

    #[inline]
    fn into_shape(self) -> Self::IntoShape {
        (self,)
    }

    #[inline]
    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(&[self])
    }
}

impl IntoShape for Vec<usize> {
    type IntoShape = DynRank;

    #[inline]
    fn into_shape(self) -> DynRank {
        DynRank::Dyn(self.into())
    }

    #[inline]
    fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
        f(&self)
    }
}

macro_rules! impl_into_shape {
    ($n:tt, $shape:ty) => {
        impl IntoShape for [usize; $n] {
            type IntoShape = $shape;

            #[inline]
            fn into_shape(self) -> Self::IntoShape {
                Shape::from_dims(&self)
            }

            #[inline]
            fn into_dims<T, F: FnOnce(&[usize]) -> T>(self, f: F) -> T {
                f(&self)
            }
        }
    };
}

impl_into_shape!(0, ());
impl_into_shape!(1, (Dyn,));
impl_into_shape!(2, (Dyn, Dyn));
impl_into_shape!(3, (Dyn, Dyn, Dyn));
impl_into_shape!(4, (Dyn, Dyn, Dyn, Dyn));
impl_into_shape!(5, (Dyn, Dyn, Dyn, Dyn, Dyn));
impl_into_shape!(6, (Dyn, Dyn, Dyn, Dyn, Dyn, Dyn));
